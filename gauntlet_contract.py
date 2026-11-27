#!/usr/bin/env python3
"""Create and verify hash-bound (SHA-256) Gordon Gauntlet run contracts.

The runner, evaluator, reporter, protocol fingerprinter, and promotion pipeline
consume the same profile from ``battery_manifest.json``.  A run is complete
only when every result validates against the frozen hash-bound contract and a
completion receipt binds the exact result bytes via SHA-256.

Note: contracts provide tamper-evident SHA-256 content binding rather than
public-key cryptographic signatures.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import secrets
import stat
import tempfile
import time
from typing import Any
import urllib.parse


MANIFEST_SCHEMA = "gordon-battery-manifest.v1"
RUN_SCHEMA = "gordon-run.v1"
RESULT_SCHEMA = "gordon-result.v2"
COMPLETION_SCHEMA = "gordon-run-completion.v1"
SAFE_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}\Z")
HEX16 = re.compile(r"[0-9a-f]{16}\Z")
HEX32 = re.compile(r"[0-9a-f]{32}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
MAX_JSON = 16 * 1024 * 1024


class ContractError(RuntimeError):
    pass


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ContractError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _json_bytes(raw: bytes, label: str) -> Any:
    if not raw or len(raw) > MAX_JSON:
        raise ContractError(f"{label} is empty or too large")
    try:
        return json.loads(
            raw.decode("utf-8"), object_pairs_hook=_strict_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ContractError(f"non-finite JSON value in {label}: {token}")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid UTF-8 JSON in {label}") from exc


def _safe_existing(path: Path, label: str, *, directory: bool = False) -> Path:
    path = path.expanduser()
    if not path.is_absolute() or path != Path(os.path.normpath(path)):
        raise ContractError(f"{label} must be a normalized absolute path")
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise ContractError(f"cannot stat {label}: {exc}") from exc
    if stat.S_ISLNK(info.st_mode) or os.path.realpath(path) != str(path):
        raise ContractError(f"{label} must not be or traverse a symlink")
    wanted = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
    if not wanted:
        raise ContractError(f"{label} has the wrong file type")
    return path


def _read_file(path: Path, label: str, *, limit: int = MAX_JSON) -> bytes:
    path = _safe_existing(path, label)
    pre = os.lstat(path)
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(fd)
        if (pre.st_dev, pre.st_ino) != (opened.st_dev, opened.st_ino):
            raise ContractError(f"{label} changed while opening")
        chunks: list[bytes] = []
        total = 0
        while True:
            block = os.read(fd, min(1024 * 1024, limit + 1 - total))
            if not block:
                break
            chunks.append(block)
            total += len(block)
            if total > limit:
                raise ContractError(f"{label} exceeds {limit} bytes")
        post = os.fstat(fd)
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != (
                post.st_dev, post.st_ino, post.st_size, post.st_mtime_ns):
            raise ContractError(f"{label} changed while reading")
        return b"".join(chunks)
    finally:
        os.close(fd)


def sha256_file(path: Path, label: str = "file") -> str:
    h = hashlib.sha256()
    path = _safe_existing(path, label)
    pre = os.lstat(path)
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(fd)
        if (pre.st_dev, pre.st_ino) != (opened.st_dev, opened.st_ino):
            raise ContractError(f"{label} changed while opening")
        while True:
            block = os.read(fd, 8 * 1024 * 1024)
            if not block:
                break
            h.update(block)
        post = os.fstat(fd)
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != (
                post.st_dev, post.st_ino, post.st_size, post.st_mtime_ns):
            raise ContractError(f"{label} changed while hashing")
    finally:
        os.close(fd)
    return h.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _atomic_create(path: Path, value: Any) -> None:
    if not path.is_absolute() or path != Path(os.path.normpath(path)):
        raise ContractError("output must be a normalized absolute path")
    parent = _safe_existing(path.parent, "output parent", directory=True)
    if os.path.lexists(path):
        raise ContractError(f"refusing to overwrite {path}")
    payload = canonical_bytes(value)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(tmp, 0o644)
        try:
            os.link(tmp, path)
        except FileExistsError as exc:
            raise ContractError(f"output appeared concurrently: {path}") from exc
        dfd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def _token(value: Any, label: str) -> str:
    if type(value) is not str or not SAFE_TOKEN.fullmatch(value):
        raise ContractError(f"{label} is not a safe token")
    return value


def sanitize_model_id(raw_model: Any, fallback: str = "model") -> str:
    """Normalize an arbitrary model identifier into a SAFE_TOKEN contract model_id."""
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", str(raw_model or ""))[:80] or fallback
    if not re.match(r"^[A-Za-z0-9]", safe):
        safe = "m_" + safe
    return _token(safe[:80], "model id")


def item_prompt_sha256(item: dict[str, Any], system_default: str | None = None) -> str:
    """Canonical SHA-256 digest of a battery item's effective system prompt, prompt/turns, and tools."""
    s = item.get("system", system_default)
    payload: dict[str, Any] = {
        "id": str(item.get("id", "")),
        "system": s if s else None,
        "tools": item.get("tools") or None,
    }
    if item.get("turns"):
        payload["turns"] = list(item["turns"])
    else:
        payload["prompt"] = item.get("prompt")
    return canonical_sha(payload)


def _endpoint(value: Any) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ContractError("endpoint must be a trimmed string")
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password:
        raise ContractError("endpoint must be unauthenticated HTTP(S)")
    if parsed.query or parsed.fragment or parsed.path.rstrip("/") != "/v1/chat/completions":
        raise ContractError("endpoint path must be /v1/chat/completions")
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ContractError("endpoint port is malformed") from exc
    if not parsed.hostname:
        raise ContractError("endpoint host is missing")
    return value


def load_battery_frozen(
    root: Path, name: str
) -> tuple[dict[str, Any], bytes, list[dict[str, Any]]]:
    """Read, validate, and hash a battery JSON file once; return (entry, raw_bytes, items)."""
    name = _token(name, "battery name")
    path = _safe_existing(root / "batteries" / f"{name}.json", f"battery {name}")
    raw = _read_file(path, f"battery {name}")
    value = _json_bytes(raw, f"battery {name}")
    if not isinstance(value, list) or not value:
        raise ContractError(f"battery {name} must be a non-empty list")
    ids: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ContractError(f"battery {name} item {index} is not an object")
        item_id = _token(item.get("id"), f"battery {name} item id")
        if item_id in ids:
            raise ContractError(f"battery {name} has duplicate id {item_id}")
        if "prompt" not in item and "turns" not in item:
            raise ContractError(f"battery {name}/{item_id} has no prompt or turns")
        ids.append(item_id)
    digest = hashlib.sha256(raw).hexdigest()
    entry = {
        "name": name,
        "file": f"batteries/{name}.json",
        "sha256": digest,
        "version": digest[:16],
        "ids": ids,
    }
    return entry, raw, value


def _battery_entry(root: Path, name: str) -> dict[str, Any]:
    entry, _raw, _items = load_battery_frozen(root, name)
    return entry


def load_profile_frozen(
    root: Path, profile: str
) -> tuple[str, list[dict[str, Any]], str, dict[str, dict[str, Any]]]:
    """Read, validate, and freeze a manifest profile and every battery's exact bytes."""
    root = _safe_existing(root.expanduser().absolute(), "gauntlet root", directory=True)
    profile = _token(profile, "profile")
    manifest_path = _safe_existing(root / "battery_manifest.json", "battery manifest")
    raw = _read_file(manifest_path, "battery manifest")
    manifest = _json_bytes(raw, "battery manifest")
    if not isinstance(manifest, dict) or set(manifest) != {"schema", "profiles"}:
        raise ContractError("battery manifest has missing or unexpected fields")
    if manifest.get("schema") != MANIFEST_SCHEMA or not isinstance(manifest.get("profiles"), dict):
        raise ContractError("battery manifest schema/profiles are invalid")
    profiles = manifest["profiles"]
    if not profiles:
        raise ContractError("battery manifest must define at least one profile")
    for prof_name, prof_batteries in profiles.items():
        _token(prof_name, "profile name")
        if not isinstance(prof_batteries, list) or not prof_batteries:
            raise ContractError(f"profile {prof_name} must be a non-empty list")
        prof_checked = [_token(name, "battery name") for name in prof_batteries]
        if prof_checked != sorted(prof_checked) or len(prof_checked) != len(set(prof_checked)):
            raise ContractError(f"profile {prof_name} must be sorted and unique")
    names = profiles.get(profile)
    if not isinstance(names, list) or not names:
        raise ContractError(f"unknown or empty battery profile: {profile}")
    checked = [_token(name, "battery name") for name in names]
    entries: list[dict[str, Any]] = []
    frozen: dict[str, dict[str, Any]] = {}
    for name in checked:
        entry, batt_raw, items = load_battery_frozen(root, name)
        entries.append(entry)
        frozen[name] = {
            "entry": entry,
            "raw": batt_raw,
            "sha256": entry["sha256"],
            "items": items,
        }
    battery_set_sha = canonical_sha({"profile": profile, "batteries": entries})
    return hashlib.sha256(raw).hexdigest(), entries, battery_set_sha, frozen


def load_profile(root: Path, profile: str) -> tuple[str, list[dict[str, Any]], str]:
    manifest_sha, entries, battery_set_sha, _frozen = load_profile_frozen(root, profile)
    return manifest_sha, entries, battery_set_sha


def freeze_run_manifest(
    root: Path,
    *,
    profile: str,
    label: str,
    endpoint: str,
    model_id: str,
    out_path: Path,
    system: str = "none",
    thinking: str | None = None,
    max_tokens: int | None = None,
    timeout: int = 240,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], str | None]:
    """Atomically create a hash-bound run manifest and return (run, frozen_batteries, frozen_system_text).

    Reads manifest, battery files, and optional system prompt once before any
    model calls so execution consumes the exact bytes bound in the manifest.
    """
    root = Path(root).expanduser().absolute()
    label = _token(label, "label")
    model_id = _token(model_id, "model id")
    endpoint = _endpoint(endpoint)
    profile = _token(profile, "profile")
    manifest_sha, batteries, battery_set_sha, frozen = load_profile_frozen(root, profile)
    sys_receipt: dict[str, Any] | None = None
    frozen_system_text: str | None = None
    if system and system != "none":
        system_path = _safe_existing(Path(system).expanduser().absolute(), "system prompt")
        sys_raw = _read_file(system_path, "system prompt")
        try:
            frozen_system_text = sys_raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ContractError("system prompt is not valid UTF-8") from exc
        sys_receipt = {
            "path": str(system_path),
            "size": len(sys_raw),
            "sha256": hashlib.sha256(sys_raw).hexdigest(),
        }
    value = {
        "schema": RUN_SCHEMA,
        "run_id": secrets.token_hex(16),
        "created_unix": int(time.time()),
        "label": label,
        "endpoint": endpoint,
        "model_id": model_id,
        "profile": profile,
        "manifest_sha256": manifest_sha,
        "battery_set_sha256": battery_set_sha,
        "system": sys_receipt,
        "generation": {
            "temperature": 0.0,
            "thinking": thinking,
            "max_tokens_override": max_tokens,
            "timeout_seconds": timeout,
        },
        "batteries": batteries,
    }
    _atomic_create(Path(out_path).expanduser().absolute(), value)
    return value, frozen, frozen_system_text


def create_run_manifest(
    root: Path,
    *,
    profile: str,
    label: str,
    endpoint: str,
    model_id: str,
    out_path: Path,
    system: str = "none",
    thinking: str | None = None,
    max_tokens: int | None = None,
    timeout: int = 240,
) -> dict[str, Any]:
    """Create a canonical hash-bound run manifest atomically and return its dict."""
    value, _frozen, _sys = freeze_run_manifest(
        root,
        profile=profile,
        label=label,
        endpoint=endpoint,
        model_id=model_id,
        out_path=out_path,
        system=system,
        thinking=thinking,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    return value


def write_battery_result_v2(
    run: dict[str, Any],
    battery_name: str,
    rows: list[dict[str, Any]],
    out_path: Path,
    *,
    expected_battery_sha256: str | None = None,
) -> dict[str, Any]:
    """Serialize battery rows into the exact gordon-result.v2 contract schema."""
    battery_name = _token(battery_name, "battery name")
    entry = next((item for item in run["batteries"] if item["name"] == battery_name), None)
    if entry is None:
        raise ContractError(f"battery {battery_name} is not in the run contract")
    if expected_battery_sha256 is not None and expected_battery_sha256 != entry["sha256"]:
        raise ContractError(
            f"battery {battery_name} frozen sha256 does not match run contract"
        )
    formatted_rows: list[dict[str, Any]] = []
    for row in rows:
        row_bsha = row.get("battery_sha256")
        if row_bsha is not None and row_bsha != entry["sha256"]:
            raise ContractError(
                f"battery {battery_name} row {row.get('id')} sha256 does not match run contract"
            )
        sec = float(row.get("sec", row.get("secs", 0.0)))
        ctok = int(row.get("ctok", row.get("completion_tokens", 0)))
        tps = round(ctok / sec, 2) if sec > 0 else 0.0
        item: dict[str, Any] = {
            "id": str(row["id"]),
            "sec": sec,
            "finish": row.get("finish", row.get("finish_reason")),
            "ctok": ctok,
            "tps": tps,
            "tool_calls": bool(row.get("tool_calls")),
            "err": row.get("err"),
            "text": str(row.get("text") or ""),
        }
        if "transcript" in row:
            item["transcript"] = row["transcript"]
        if "model" in row and row["model"] is not None:
            item["model"] = str(row["model"])
        if "prompt_sha256" in row and row["prompt_sha256"] is not None:
            item["prompt_sha256"] = str(row["prompt_sha256"])
        formatted_rows.append(item)
    payload = {
        "schema": RESULT_SCHEMA,
        "run_id": run["run_id"],
        "label": run["label"],
        "endpoint": run["endpoint"],
        "model_id": run["model_id"],
        "battery": f"{battery_name}.json",
        "battery_sha256": entry["sha256"],
        "battery_version": entry["version"],
        "battery_set_sha256": run["battery_set_sha256"],
        "system_sha256": None if run["system"] is None else run["system"]["sha256"],
        "generation": run["generation"],
        "results": formatted_rows,
    }
    dest = Path(out_path).expanduser().absolute()
    _atomic_create(dest, payload)
    return validate_result(run, dest, battery_name)


def complete_run_manifest(
    root: Path,
    run_path: Path,
    results_dir: Path,
    out_path: Path,
) -> dict[str, Any]:
    """Validate every battery result and atomically emit a completion receipt."""
    root = Path(root).expanduser().absolute()
    run_path = Path(run_path).expanduser().absolute()
    run, raw = load_run(run_path, root=root, revalidate_inputs=True)
    res_dir = _safe_existing(Path(results_dir).expanduser().absolute(),
                             "results directory", directory=True)
    results = []
    for entry in run["batteries"]:
        path = res_dir / f"{entry['name']}.{run['label']}.json"
        validate_result(run, path, entry["name"], root=root)
        results.append({"battery": entry["name"], "sha256": sha256_file(path, "result")})
    value = {
        "schema": COMPLETION_SCHEMA,
        "run_id": run["run_id"],
        "label": run["label"],
        "run_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "battery_set_sha256": run["battery_set_sha256"],
        "results": results,
    }
    _atomic_create(Path(out_path).expanduser().absolute(), value)
    return value


def create_run(args: argparse.Namespace) -> None:
    value = create_run_manifest(
        Path(args.root),
        profile=args.profile,
        label=args.label,
        endpoint=args.endpoint,
        model_id=args.model_id,
        out_path=Path(args.out),
        system=args.system,
        thinking=args.thinking,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )
    print(value["run_id"])


def load_run(path: Path, *, root: Path | None = None,
             revalidate_inputs: bool = True) -> tuple[dict[str, Any], bytes]:
    raw = _read_file(path.expanduser().absolute(), "run manifest")
    value = _json_bytes(raw, "run manifest")
    keys = {"schema", "run_id", "created_unix", "label", "endpoint", "model_id",
            "profile", "manifest_sha256", "battery_set_sha256", "system",
            "generation", "batteries"}
    if not isinstance(value, dict) or set(value) != keys:
        raise ContractError("run manifest has missing or unexpected fields")
    if raw != canonical_bytes(value):
        raise ContractError("run manifest is not canonical JSON")
    if value.get("schema") != RUN_SCHEMA or not HEX32.fullmatch(str(value.get("run_id", ""))):
        raise ContractError("run manifest schema/run_id is invalid")
    _token(value.get("label"), "run label")
    _token(value.get("model_id"), "run model id")
    _token(value.get("profile"), "run profile")
    _endpoint(value.get("endpoint"))
    if type(value.get("created_unix")) is not int or value["created_unix"] <= 0:
        raise ContractError("run created_unix is invalid")
    if not HEX64.fullmatch(str(value.get("manifest_sha256", ""))) or not HEX64.fullmatch(
            str(value.get("battery_set_sha256", ""))):
        raise ContractError("run fingerprint is malformed")
    generation = value.get("generation")
    if not isinstance(generation, dict) or set(generation) != {
            "temperature", "thinking", "max_tokens_override", "timeout_seconds"}:
        raise ContractError("run generation config is invalid")
    if generation["temperature"] != 0.0 or type(generation["timeout_seconds"]) is not int \
            or generation["timeout_seconds"] <= 0:
        raise ContractError("run generation settings are invalid")
    if generation["thinking"] not in {None, "enabled", "disabled", "adaptive"}:
        raise ContractError("run thinking setting is invalid")
    if generation["max_tokens_override"] is not None and (
            type(generation["max_tokens_override"]) is not int
            or generation["max_tokens_override"] <= 0):
        raise ContractError("run max token override is invalid")
    if revalidate_inputs:
        if root is None:
            raise ContractError("root is required to revalidate run inputs")
        manifest_sha, batteries, battery_set_sha = load_profile(root, value["profile"])
        if value["manifest_sha256"] != manifest_sha or value["batteries"] != batteries \
                or value["battery_set_sha256"] != battery_set_sha:
            raise ContractError("battery inputs changed after run contract creation")
        system = value.get("system")
        if system is not None:
            if not isinstance(system, dict) or set(system) != {"path", "size", "sha256"}:
                raise ContractError("run system prompt receipt is malformed")
            system_path = _safe_existing(Path(system["path"]), "system prompt")
            if system["size"] != system_path.stat().st_size or system["sha256"] != sha256_file(
                    system_path, "system prompt"):
                raise ContractError("system prompt changed after run contract creation")
    return value, raw


def validate_result(
    run: dict[str, Any],
    result_path: Path,
    battery_name: str,
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    battery_name = _token(battery_name, "battery name")
    entry = next((item for item in run["batteries"] if item["name"] == battery_name), None)
    if entry is None:
        raise ContractError(f"battery {battery_name} is not in the run contract")
    raw = _read_file(result_path.expanduser().absolute(), f"result {battery_name}")
    value = _json_bytes(raw, f"result {battery_name}")
    keys = {"schema", "run_id", "label", "endpoint", "model_id", "battery",
            "battery_sha256", "battery_version", "battery_set_sha256",
            "system_sha256", "generation", "results"}
    if not isinstance(value, dict) or set(value) != keys:
        raise ContractError(f"result {battery_name} has missing or unexpected fields")
    expected = {
        "schema": RESULT_SCHEMA,
        "run_id": run["run_id"],
        "label": run["label"],
        "endpoint": run["endpoint"],
        "model_id": run["model_id"],
        "battery": f"{battery_name}.json",
        "battery_sha256": entry["sha256"],
        "battery_version": entry["version"],
        "battery_set_sha256": run["battery_set_sha256"],
        "system_sha256": None if run["system"] is None else run["system"]["sha256"],
        "generation": run["generation"],
    }
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise ContractError(f"result {battery_name} {key} does not match run contract")
    expected_prompt_by_id: dict[str, str] | None = None
    if root is not None:
        root_dir = _safe_existing(root.expanduser().absolute(), "gauntlet root", directory=True)
        disk_entry, _disk_raw, disk_items = load_battery_frozen(root_dir, battery_name)
        if disk_entry["sha256"] != entry["sha256"]:
            raise ContractError(f"battery {battery_name} changed on disk after run contract creation")
        sys_text: str | None = None
        if run.get("system") is not None:
            sys_path = _safe_existing(Path(run["system"]["path"]), "system prompt")
            sys_raw = _read_file(sys_path, "system prompt")
            if hashlib.sha256(sys_raw).hexdigest() != run["system"]["sha256"]:
                raise ContractError("system prompt changed after run contract creation")
            sys_text = sys_raw.decode("utf-8")
        expected_prompt_by_id = {
            str(it["id"]): item_prompt_sha256(it, system_default=sys_text)
            for it in disk_items
        }
    results = value.get("results")
    if not isinstance(results, list) or len(results) != len(entry["ids"]):
        raise ContractError(f"result {battery_name} row count is wrong")
    got_ids = []
    base_keys = frozenset({"id", "sec", "finish", "ctok", "tps", "tool_calls", "err", "text"})
    allowed_keys = base_keys | frozenset({"transcript", "model", "prompt_sha256"})
    for row in results:
        row_keys = frozenset(row) if isinstance(row, dict) else frozenset()
        if not isinstance(row, dict) or not (base_keys <= row_keys <= allowed_keys):
            raise ContractError(f"result {battery_name} row schema is invalid")
        row_id = _token(row.get("id"), "result id")
        got_ids.append(row_id)
        if type(row.get("sec")) not in {int, float} or isinstance(row.get("sec"), bool) \
                or row["sec"] < 0:
            raise ContractError(f"result {battery_name} has invalid latency")
        if type(row.get("ctok")) is not int or isinstance(row.get("ctok"), bool) \
                or row["ctok"] < 0 or type(row.get("tool_calls")) is not bool:
            raise ContractError(f"result {battery_name} row counters are invalid")
        if type(row.get("text")) is not str or (row.get("err") is not None
                                                 and type(row.get("err")) is not str):
            raise ContractError(f"result {battery_name} row text/error is invalid")
        if "model" in row:
            if type(row.get("model")) is not str or not row["model"].strip():
                raise ContractError(f"result {battery_name} row model is invalid")
            if sanitize_model_id(row["model"], run["label"]) != run["model_id"]:
                raise ContractError(
                    f"result {battery_name}/{row_id} effective model does not match run contract"
                )
        if "prompt_sha256" in row:
            psha = row.get("prompt_sha256")
            if type(psha) is not str or not HEX64.fullmatch(psha):
                raise ContractError(f"result {battery_name}/{row_id} prompt_sha256 is malformed")
            if expected_prompt_by_id is not None and row.get("err") is None:
                if expected_prompt_by_id.get(row_id) != psha:
                    raise ContractError(
                        f"result {battery_name}/{row_id} prompt_sha256 does not match frozen battery"
                    )
    if got_ids != entry["ids"] or len(got_ids) != len(set(got_ids)):
        raise ContractError(f"result {battery_name} IDs differ from the frozen battery")
    return value


def complete_run(args: argparse.Namespace) -> None:
    value = complete_run_manifest(
        Path(args.root),
        Path(args.run_manifest),
        Path(args.results_dir),
        Path(args.out),
    )
    print(value["run_id"])


def verify_complete(root: Path, run_path: Path, completion_path: Path,
                    results_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    run, run_raw = load_run(run_path, root=root, revalidate_inputs=True)
    comp_raw = _read_file(completion_path.expanduser().absolute(), "completion receipt")
    comp = _json_bytes(comp_raw, "completion receipt")
    if comp_raw != canonical_bytes(comp):
        raise ContractError("completion receipt is not canonical JSON")
    keys = {"schema", "run_id", "label", "run_manifest_sha256",
            "battery_set_sha256", "results"}
    if not isinstance(comp, dict) or set(comp) != keys:
        raise ContractError("completion receipt has missing or unexpected fields")
    expected = {
        "schema": COMPLETION_SCHEMA,
        "run_id": run["run_id"],
        "label": run["label"],
        "run_manifest_sha256": hashlib.sha256(run_raw).hexdigest(),
        "battery_set_sha256": run["battery_set_sha256"],
    }
    for key, wanted in expected.items():
        if comp.get(key) != wanted:
            raise ContractError(f"completion {key} does not match run")
    rows = comp.get("results")
    expected_names = [entry["name"] for entry in run["batteries"]]
    if not isinstance(rows, list) or [row.get("battery") for row in rows
                                     if isinstance(row, dict)] != expected_names:
        raise ContractError("completion result set/order is wrong")
    loaded: dict[str, dict[str, Any]] = {}
    for row in rows:
        if set(row) != {"battery", "sha256"} or not HEX64.fullmatch(str(row.get("sha256", ""))):
            raise ContractError("completion result fingerprint is malformed")
        path = results_dir / f"{row['battery']}.{run['label']}.json"
        if sha256_file(path, "result") != row["sha256"]:
            raise ContractError(f"result {row['battery']} changed after completion")
        loaded[row["battery"]] = validate_result(run, path, row["battery"], root=root)
    return run, loaded


def verify_cmd(args: argparse.Namespace) -> None:
    run, _ = verify_complete(
        Path(args.root).expanduser().absolute(),
        Path(args.run_manifest).expanduser().absolute(),
        Path(args.completion).expanduser().absolute(),
        Path(args.results_dir).expanduser().absolute(),
    )
    print(run["run_id"])


def list_cmd(args: argparse.Namespace) -> None:
    _manifest_sha, entries, _set_sha = load_profile(
        Path(args.root).expanduser().absolute(), args.profile)
    for entry in entries:
        print(entry["name"])


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    ls = sub.add_parser("list")
    ls.add_argument("--root", required=True)
    ls.add_argument("--profile", default="standard")
    ls.set_defaults(func=list_cmd)
    create = sub.add_parser("create-run")
    create.add_argument("--root", required=True)
    create.add_argument("--profile", default="standard")
    create.add_argument("--label", required=True)
    create.add_argument("--endpoint", required=True)
    create.add_argument("--model-id", required=True)
    create.add_argument("--system", default="none")
    create.add_argument("--thinking", choices=("enabled", "disabled", "adaptive"))
    create.add_argument("--max-tokens", type=int)
    create.add_argument("--timeout", type=int, default=240)
    create.add_argument("--out", required=True)
    create.set_defaults(func=create_run)
    complete = sub.add_parser("complete-run")
    complete.add_argument("--root", required=True)
    complete.add_argument("--run-manifest", required=True)
    complete.add_argument("--results-dir", required=True)
    complete.add_argument("--out", required=True)
    complete.set_defaults(func=complete_run)
    verify = sub.add_parser("verify-run")
    verify.add_argument("--root", required=True)
    verify.add_argument("--run-manifest", required=True)
    verify.add_argument("--completion", required=True)
    verify.add_argument("--results-dir", required=True)
    verify.set_defaults(func=verify_cmd)
    return p


def main() -> int:
    args = parser().parse_args()
    try:
        args.func(args)
    except (ContractError, OSError, ValueError) as exc:
        print(f"GAUNTLET CONTRACT FAIL: {exc}", file=os.sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
