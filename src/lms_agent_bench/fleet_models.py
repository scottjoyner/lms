#!/usr/bin/env python3
"""Discover local model artifacts and fingerprint only the models that matter.

The scanner is intentionally backend-neutral. It records enough metadata for
``lms-fleet plan`` without forcing a full multi-gigabyte SHA-256 pass across
every model. A selected model can later be upgraded to a full content hash
before a desired profile is imported.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from lms_agent_bench.fleet_loadout import canonical_hash, load_json, write_json

SCHEMA_VERSION = "fleet_model_inventory.v1"
DEFAULT_EXTENSIONS = (".gguf", ".onnx", ".safetensors")
QUICK_SAMPLE_BYTES = 1024 * 1024


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def normalize_sha256(value: str) -> str:
    value = value.strip()
    return value if value.startswith("sha256:") else f"sha256:{value}"


def hash_file(path: Path, mode: str, chunk_size: int = 8 * 1024 * 1024) -> Optional[str]:
    if mode == "none":
        return None
    digest = hashlib.sha256()
    size = path.stat().st_size
    if mode == "full":
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(chunk_size)
                if not chunk:
                    break
                digest.update(chunk)
        return normalize_sha256(digest.hexdigest())
    if mode != "quick":
        raise ValueError(f"unsupported hash mode: {mode}")
    digest.update(str(size).encode("ascii"))
    digest.update(b"\0")
    with path.open("rb") as handle:
        digest.update(handle.read(QUICK_SAMPLE_BYTES))
        if size > QUICK_SAMPLE_BYTES:
            handle.seek(max(0, size - QUICK_SAMPLE_BYTES))
            digest.update(handle.read(QUICK_SAMPLE_BYTES))
    return normalize_sha256(digest.hexdigest())


def artifact_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    return sorted(item for item in path.rglob("*") if item.is_file())


def artifact_size_and_mtime(path: Path) -> tuple[int, int]:
    files = artifact_files(path)
    if not files:
        stat = path.stat()
        return 0, stat.st_mtime_ns
    return sum(item.stat().st_size for item in files), max(item.stat().st_mtime_ns for item in files)


def hash_artifact(path: Path, mode: str) -> Optional[str]:
    if path.is_file():
        return hash_file(path, mode)
    if mode == "none":
        return None
    digest = hashlib.sha256()
    for item in artifact_files(path):
        relative = str(item.relative_to(path)).replace(os.sep, "/")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(item.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        if mode == "full":
            value = hash_file(item, "full")
        elif mode == "quick":
            value = hash_file(item, "quick")
        else:
            raise ValueError(f"unsupported hash mode: {mode}")
        digest.update(str(value).encode("ascii"))
        digest.update(b"\0")
    return normalize_sha256(digest.hexdigest())


def parse_quantization(name: str) -> str:
    patterns = (
        r"(?i)(?:^|[-_.])(IQ\d(?:_[A-Z0-9]+)*)(?:[-_.]|$)",
        r"(?i)(?:^|[-_.])(Q\d(?:_[A-Z0-9]+)*)(?:[-_.]|$)",
        r"(?i)(?:^|[-_.])((?:BF|F)(?:16|32))(?:[-_.]|$)",
        r"(?i)(?:^|[-_.])(INT(?:4|8))(?:[-_.]|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            return match.group(1).upper()
    return "unknown"


def parse_parameter_billions(name: str) -> Optional[float]:
    matches = re.findall(r"(?i)(?:^|[-_.])(\d+(?:\.\d+)?)B(?:[-_.]|$)", name)
    if not matches:
        return None
    values = [float(item) for item in matches]
    plausible = [item for item in values if 0.05 <= item <= 1000]
    return max(plausible) if plausible else None


def is_model_directory(path: Path) -> bool:
    if not path.is_dir():
        return False
    has_config = (path / "genai_config.json").is_file() or (path / "config.json").is_file()
    has_model = any(path.glob("*.onnx"))
    has_tokenizer = (path / "tokenizer.json").is_file() or (path / "tokenizer.model").is_file()
    return has_config and has_model and has_tokenizer


def iter_model_paths(roots: Sequence[str], extensions: Sequence[str]) -> Iterable[Path]:
    allowed = {item.lower() if item.startswith(".") else f".{item.lower()}" for item in extensions}
    seen: set[str] = set()
    model_directories: List[Path] = []
    for raw_root in roots:
        root = Path(raw_root).expanduser().resolve()
        if root.is_dir():
            if is_model_directory(root):
                model_directories.append(root)
            for marker in root.rglob("genai_config.json"):
                if is_model_directory(marker.parent):
                    model_directories.append(marker.parent.resolve())
    unique_directories = sorted({item.resolve() for item in model_directories})
    for directory in unique_directories:
        key = str(directory)
        if key not in seen:
            seen.add(key)
            yield directory
    for raw_root in roots:
        root = Path(raw_root).expanduser().resolve()
        if root.is_file():
            candidates: Iterable[Path] = [root]
        elif root.is_dir():
            candidates = root.rglob("*")
        else:
            continue
        for candidate in candidates:
            if not candidate.is_file() or candidate.suffix.lower() not in allowed:
                continue
            resolved = candidate.resolve()
            if any(directory == resolved.parent or directory in resolved.parents for directory in unique_directories):
                continue
            key = str(resolved)
            if key in seen:
                continue
            seen.add(key)
            yield resolved


def stable_model_id(path: Path, used: set[str]) -> str:
    base = path.name
    if base not in used:
        used.add(base)
        return base
    suffix = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:8]
    value = f"{base}@{suffix}"
    used.add(value)
    return value


def record_model(
    path: Path,
    model_id: str,
    hash_mode: str,
    default_max_context: int,
) -> Dict[str, Any]:
    size_bytes, mtime_ns = artifact_size_and_mtime(path)
    fingerprint = hash_artifact(path, hash_mode)
    record: Dict[str, Any] = {
        "id": model_id,
        "path": str(path),
        "format": "onnx-genai" if path.is_dir() else path.suffix.lower().lstrip("."),
        "size_bytes": size_bytes,
        "mtime_ns": mtime_ns,
        "quantization": parse_quantization(path.name),
        "parameter_billions": parse_parameter_billions(path.name),
        "max_context": default_max_context,
        "fingerprint_mode": hash_mode,
    }
    if hash_mode == "full" and fingerprint:
        record["artifact_fingerprint"] = fingerprint
        record["content_sha256"] = fingerprint
    elif hash_mode == "quick" and fingerprint:
        record["quick_fingerprint"] = fingerprint
    return record


def scan_model_roots(
    roots: Sequence[str],
    hash_mode: str = "quick",
    default_max_context: int = 32768,
    extensions: Sequence[str] = DEFAULT_EXTENSIONS,
) -> Dict[str, Any]:
    if not roots:
        raise ValueError("at least one model root is required")
    if default_max_context <= 0:
        raise ValueError("default_max_context must be positive")
    used: set[str] = set()
    models = [
        record_model(path, stable_model_id(path, used), hash_mode, default_max_context)
        for path in iter_model_paths(roots, extensions)
    ]
    models.sort(key=lambda item: (str(item["id"]).lower(), str(item["path"])))
    core = {
        "roots": [str(Path(item).expanduser()) for item in roots],
        "hash_mode": hash_mode,
        "models": models,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "model_inventory",
        "created_at_utc": utc_now_iso(),
        **core,
        "inventory_fingerprint": canonical_hash(core),
        "authority": {
            "kind": "observation",
            "may_admit_runtime": False,
            "full_content_hash_required_for_profile": True,
        },
    }


def selection_model_id(selection: Mapping[str, Any]) -> str:
    selected = selection.get("selected")
    if not isinstance(selected, Mapping):
        raise ValueError("selection has no selected candidate")
    candidate = selected.get("candidate")
    if not isinstance(candidate, Mapping):
        raise ValueError("selection.selected has no candidate")
    model = candidate.get("model")
    if not isinstance(model, Mapping) or not model.get("id"):
        raise ValueError("selected candidate has no model id")
    return str(model["id"])


def fingerprint_inventory(
    inventory: Mapping[str, Any],
    model_ids: Sequence[str],
) -> Dict[str, Any]:
    requested = set(model_ids)
    if not requested:
        raise ValueError("at least one model id is required")
    output = json.loads(json.dumps(inventory))
    models = output.get("models", [])
    found: set[str] = set()
    for model in models:
        if not isinstance(model, dict) or str(model.get("id")) not in requested:
            continue
        path = Path(str(model.get("path", ""))).expanduser()
        if not path.exists() or not (path.is_file() or path.is_dir()):
            raise ValueError(f"selected model path does not exist: {path}")
        fingerprint = hash_artifact(path, "full")
        size_bytes, mtime_ns = artifact_size_and_mtime(path)
        model["artifact_fingerprint"] = fingerprint
        model["content_sha256"] = fingerprint
        model["fingerprint_mode"] = "full"
        model["size_bytes"] = size_bytes
        model["mtime_ns"] = mtime_ns
        found.add(str(model["id"]))
    missing = sorted(requested - found)
    if missing:
        raise ValueError(f"model ids not present in inventory: {', '.join(missing)}")
    core = {
        "roots": output.get("roots", []),
        "hash_mode": "mixed",
        "models": models,
    }
    output.update(
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "model_inventory",
            "updated_at_utc": utc_now_iso(),
            "hash_mode": "mixed",
            "models": models,
            "inventory_fingerprint": canonical_hash(core),
        }
    )
    return output


def parse_extensions(value: str) -> List[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("extensions cannot be empty")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Discover and fingerprint local model artifacts")
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="Scan model roots and create a planning inventory")
    scan.add_argument("--root", action="append", required=True)
    scan.add_argument("--hash-mode", choices=("none", "quick", "full"), default="quick")
    scan.add_argument("--default-max-context", type=int, default=32768)
    scan.add_argument("--extensions", default=",".join(DEFAULT_EXTENSIONS))
    scan.add_argument("--out", default="model_inventory.json")

    fingerprint = sub.add_parser(
        "fingerprint",
        help="Upgrade selected inventory entries to full content SHA-256 hashes",
    )
    fingerprint.add_argument("--inventory", required=True)
    fingerprint.add_argument("--model-id", action="append", default=[])
    fingerprint.add_argument("--selection", default=None)
    fingerprint.add_argument("--out", default="model_inventory.fingerprinted.json")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "scan":
        artifact = scan_model_roots(
            args.root,
            hash_mode=args.hash_mode,
            default_max_context=args.default_max_context,
            extensions=parse_extensions(args.extensions),
        )
    elif args.command == "fingerprint":
        model_ids = list(args.model_id)
        if args.selection:
            model_ids.append(selection_model_id(load_json(args.selection)))
        artifact = fingerprint_inventory(load_json(args.inventory), model_ids)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    write_json(args.out, artifact)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
