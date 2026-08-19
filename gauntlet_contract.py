#!/usr/bin/env python3
"""Create and verify immutable Gordon Gauntlet run contracts.

The runner, evaluator, reporter, protocol fingerprinter, and promotion pipeline
all consume the same profile from ``battery_manifest.json``.  A run is complete
only when every result validates against the frozen contract and a completion
receipt binds the exact result bytes.
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


def _battery_entry(root: Path, name: str) -> dict[str, Any]:
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
    return {
        "name": name,
        "file": f"batteries/{name}.json",
        "sha256": digest,
        "version": digest[:16],
        "ids": ids,
    }


def load_profile(root: Path, profile: str) -> tuple[str, list[dict[str, Any]], str]:
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
    if set(profiles) != {"standard", "heavy"}:
        raise ContractError("battery manifest must define exactly standard and heavy")
    names = profiles.get(profile)
    if not isinstance(names, list) or not names:
        raise ContractError(f"unknown or empty battery profile: {profile}")
    checked = [_token(name, "battery name") for name in names]
    if checked != sorted(checked) or len(checked) != len(set(checked)):
        raise ContractError(f"profile {profile} must be sorted and unique")
    entries = [_battery_entry(root, name) for name in checked]
    battery_set_sha = canonical_sha({"profile": profile, "batteries": entries})
    return hashlib.sha256(raw).hexdigest(), entries, battery_set_sha


def create_run(args: argparse.Namespace) -> None:
    root = Path(args.root).expanduser().absolute()
    label = _token(args.label, "label")
    model_id = _token(args.model_id, "model id")
    endpoint = _endpoint(args.endpoint)
    profile = _token(args.profile, "profile")
    manifest_sha, batteries, battery_set_sha = load_profile(root, profile)
    system: dict[str, Any] | None = None
    if args.system != "none":
        system_path = _safe_existing(Path(args.system).expanduser().absolute(), "system prompt")
        system = {
            "path": str(system_path),
            "size": system_path.stat().st_size,
            "sha256": sha256_file(system_path, "system prompt"),
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
        "system": system,
        "generation": {
            "temperature": 0.0,
            "thinking": args.thinking,
            "max_tokens_override": args.max_tokens,
            "timeout_seconds": args.timeout,
        },
        "batteries": batteries,
    }
    _atomic_create(Path(args.out).expanduser().absolute(), value)
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


def validate_result(run: dict[str, Any], result_path: Path, battery_name: str) -> dict[str, Any]:
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
    results = value.get("results")
    if not isinstance(results, list) or len(results) != len(entry["ids"]):
        raise ContractError(f"result {battery_name} row count is wrong")
    got_ids = []
    base_keys = {"id", "sec", "finish", "ctok", "tps", "tool_calls", "err", "text"}
    for row in results:
        row_keys = frozenset(row) if isinstance(row, dict) else frozenset()
        if not isinstance(row, dict) or row_keys not in {frozenset(base_keys),
                                                         frozenset(base_keys | {"transcript"})}:
            raise ContractError(f"result {battery_name} row schema is invalid")
        got_ids.append(_token(row.get("id"), "result id"))
        if type(row.get("sec")) not in {int, float} or isinstance(row.get("sec"), bool) \
                or row["sec"] < 0:
            raise ContractError(f"result {battery_name} has invalid latency")
        if type(row.get("ctok")) is not int or isinstance(row.get("ctok"), bool) \
                or row["ctok"] < 0 or type(row.get("tool_calls")) is not bool:
            raise ContractError(f"result {battery_name} row counters are invalid")
        if type(row.get("text")) is not str or (row.get("err") is not None
                                                 and type(row.get("err")) is not str):
            raise ContractError(f"result {battery_name} row text/error is invalid")
    if got_ids != entry["ids"] or len(got_ids) != len(set(got_ids)):
        raise ContractError(f"result {battery_name} IDs differ from the frozen battery")
    return value


def complete_run(args: argparse.Namespace) -> None:
    root = Path(args.root).expanduser().absolute()
    run_path = Path(args.run_manifest).expanduser().absolute()
    run, raw = load_run(run_path, root=root, revalidate_inputs=True)
    results_dir = _safe_existing(Path(args.results_dir).expanduser().absolute(),
                                 "results directory", directory=True)
    results = []
    for entry in run["batteries"]:
        path = results_dir / f"{entry['name']}.{run['label']}.json"
        validate_result(run, path, entry["name"])
        results.append({"battery": entry["name"], "sha256": sha256_file(path, "result")})
    value = {
        "schema": COMPLETION_SCHEMA,
        "run_id": run["run_id"],
        "label": run["label"],
        "run_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "battery_set_sha256": run["battery_set_sha256"],
        "results": results,
    }
    _atomic_create(Path(args.out).expanduser().absolute(), value)
    print(run["run_id"])


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
        loaded[row["battery"]] = validate_result(run, path, row["battery"])
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
    ls.add_argument("--profile", default="standard", choices=("standard", "heavy"))
    ls.set_defaults(func=list_cmd)
    create = sub.add_parser("create-run")
    create.add_argument("--root", required=True)
    create.add_argument("--profile", default="standard", choices=("standard", "heavy"))
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
