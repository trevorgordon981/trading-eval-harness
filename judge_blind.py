#!/usr/bin/env python3
"""judge_blind.py — shared helpers to make the in-session LLM-judge BLIND.

The judge must never see which MODEL produced an answer (identity bias). Every
scorer (gg_domains.py, trading_v2/p2_wrap_314.py, report.py) routes its
unresolved llm_judge items through here:

  * each item's key, rubric, prompt, response, and protocol context are hashed
    to an OPAQUE `answer_uid`; a changed answer can never reuse an old grade;
  * the record written to results/pending_judge.jsonl carries ONLY the uid +
    the judge context (domain/category/criteria/prompt/response) — NEVER the
    model/label;
  * a SIDECAR map results/judge_uid_map.json ({uid: {key, model, ...}}) — which
    the judge never reads — lets the scorer resolve uid -> score afterwards;
  * records are SHUFFLED so ordering leaks nothing.

The judge grades into results/judgments.json keyed BY answer_uid ({uid: 0|1|2}).
Scorers call uid_for_record(record) to look the grade back up. 100% stdlib.
"""
import hashlib
import json
import os
import random
import re
import tempfile

# fields that identify the producing model — MUST be stripped from blind records
_LEAK = {"model", "label", "key"}
_UID = re.compile(r"u[0-9a-f]{32}\Z")


class JudgmentError(RuntimeError):
    pass


def _strict_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise JudgmentError("duplicate JSON key: %s" % key)
        value[key] = item
    return value


def _canonical_context(record):
    """Return every byte that can change the answer's grading meaning."""
    return {
        "schema": "gordon-judge-answer.v2",
        "key": str(record.get("key") or ""),
        "criteria": str(record.get("criteria") or ""),
        "prompt": str(record.get("prompt") or ""),
        "response": str(record.get("response") or ""),
        "protocol": str(record.get("protocol") or ""),
    }


def uid_for(key, *, criteria="", prompt="", response="", protocol=""):
    """Opaque ID bound to identity, grading inputs, answer, and protocol."""
    return uid_for_record({"key": key, "criteria": criteria, "prompt": prompt,
                           "response": response, "protocol": protocol})


def uid_for_record(record):
    context = _canonical_context(record)
    if not context["key"]:
        raise JudgmentError("judge record key is missing")
    raw = json.dumps(context, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=True, allow_nan=False).encode("ascii")
    return "u" + hashlib.sha256(raw).hexdigest()[:32]


def _atomic_json(path, value):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    if os.path.islink(path):
        raise JudgmentError("refusing symlink output: %s" % path)
    fd, tmp = tempfile.mkstemp(prefix=".judge.", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=1, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
        dfd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _atomic_jsonl(path, records):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    if os.path.islink(path):
        raise JudgmentError("refusing symlink output: %s" % path)
    fd, tmp = tempfile.mkstemp(prefix=".pending-judge.", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, sort_keys=True, separators=(",", ":"),
                                        ensure_ascii=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
        dfd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_pending(records, pending_path, map_path, append=False, seed=None):
    """Write BLIND pending-judge records + update the uid->identity sidecar map.

    records: list of dicts, each with a 'key' plus judge context. Returns count.
    Blind record = record minus {model,label,key} plus {answer_uid}. Shuffled.
    """
    umap = {}
    if os.path.exists(map_path):
        with open(map_path, "r", encoding="utf-8") as stream:
            umap = json.load(stream, object_pairs_hook=_strict_object)
        if not isinstance(umap, dict):
            raise JudgmentError("judge UID map root must be an object")
        for uid, identity in umap.items():
            if not _UID.fullmatch(uid) or not isinstance(identity, dict):
                raise JudgmentError("judge UID map entry is malformed")
    blind = []
    for r in records:
        key = r["key"]
        uid = uid_for_record(r)
        umap[uid] = {k: r.get(k) for k in ("key", "model", "label", "domain",
                                           "battery", "id", "category")
                     if r.get(k) is not None}
        b = {k: v for k, v in r.items() if k not in _LEAK}
        b["answer_uid"] = uid
        blind.append(b)
    existing = []
    if append and os.path.exists(pending_path):
        with open(pending_path, "r", encoding="utf-8") as stream:
            existing = [json.loads(line, object_pairs_hook=_strict_object)
                        for line in stream if line.strip()]
        for record in existing:
            if not isinstance(record, dict) or not _UID.fullmatch(
                    str(record.get("answer_uid", ""))) or any(
                        key in record for key in _LEAK):
                raise JudgmentError("existing blind pending record is malformed")
            if record["answer_uid"] not in umap:
                raise JudgmentError("existing pending record is absent from UID map")
    combined = {}
    for record in existing + blind:
        combined[record["answer_uid"]] = record
    shuffled = list(combined.values())
    (random.SystemRandom() if seed is None else random.Random(seed)).shuffle(shuffled)

    # Publish the identity sidecar first. A crash can leave harmless extra map
    # entries, but can never leave a judge packet whose identities are absent.
    _atomic_json(map_path, umap)
    _atomic_jsonl(pending_path, shuffled)
    return len(blind)


def load_judgments(path):
    """{answer_uid: 0|1|2}. Empty dict if absent/unreadable."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as stream:
        value = json.load(stream, object_pairs_hook=_strict_object,
                          parse_constant=lambda token: (_ for _ in ()).throw(
                              JudgmentError("non-finite judgment: %s" % token)))
    if not isinstance(value, dict):
        raise JudgmentError("judgments root must be an object")
    for uid, score in value.items():
        if not _UID.fullmatch(uid) or type(score) is not int or score not in {0, 1, 2}:
            raise JudgmentError("malformed judgment entry: %r" % uid)
    return value
