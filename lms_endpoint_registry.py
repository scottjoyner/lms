#!/usr/bin/env python3
"""Persistent endpoint registry for lms-bench agents.

Agents should not need to remember long --endpoint arguments. This registry keeps
known LM Studio OpenAI-compatible endpoints in a small local JSON file and can
export them to a benchmark inventory by probing /v1/models.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import socket
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


DEFAULT_REGISTRY = Path(os.environ.get("LMS_BENCH_ENDPOINTS", "~/.config/lms-bench/endpoints.json")).expanduser()


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def normalize_base_url(value: str) -> str:
    value = value.strip().rstrip("/")
    if not value.startswith("http://") and not value.startswith("https://"):
        value = "http://" + value
    if not value.endswith("/v1"):
        value += "/v1"
    return value


def registry_path(path: Optional[str]) -> Path:
    return Path(path).expanduser() if path else DEFAULT_REGISTRY


def load_registry(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"schema_version": "lms_endpoint_registry.v1", "created_at_utc": utc_now_iso(), "updated_at_utc": utc_now_iso(), "endpoints": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("endpoints"), list):
            return data
    except Exception:
        pass
    return {"schema_version": "lms_endpoint_registry.v1", "created_at_utc": utc_now_iso(), "updated_at_utc": utc_now_iso(), "endpoints": []}


def save_registry(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data["updated_at_utc"] = utc_now_iso()
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def split_tags(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return sorted({x.strip() for x in raw.split(",") if x.strip()})


def local_host_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def probe_endpoint(base_url: str, timeout: int = 8) -> Dict[str, Any]:
    url = normalize_base_url(base_url).rstrip("/") + "/models"
    started = dt.datetime.now(dt.timezone.utc)
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read(5_000_000).decode("utf-8", errors="replace")
            data = json.loads(body)
            models = []
            if isinstance(data, dict) and isinstance(data.get("data"), list):
                models = [str(m["id"]) for m in data["data"] if isinstance(m, dict) and m.get("id")]
            elapsed = (dt.datetime.now(dt.timezone.utc) - started).total_seconds()
            return {"reachable": True, "status": getattr(resp, "status", None), "elapsed_s": round(elapsed, 4), "models": models, "model_count": len(models), "error": None}
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        elapsed = (dt.datetime.now(dt.timezone.utc) - started).total_seconds()
        return {"reachable": False, "status": None, "elapsed_s": round(elapsed, 4), "models": [], "model_count": 0, "error": repr(exc)}


def select_endpoints(data: Dict[str, Any], names: Optional[Sequence[str]], tags: Optional[Sequence[str]], enabled_only: bool = True) -> List[Dict[str, Any]]:
    endpoints = list(data.get("endpoints") or [])
    if enabled_only:
        endpoints = [e for e in endpoints if e.get("enabled", True)]
    if names:
        wanted = set(names)
        endpoints = [e for e in endpoints if e.get("name") in wanted]
    if tags:
        wanted_tags = set(tags)
        endpoints = [e for e in endpoints if wanted_tags.intersection(set(e.get("tags") or []))]
    return endpoints


def cmd_add(args: argparse.Namespace) -> int:
    path = registry_path(args.registry)
    data = load_registry(path)
    name = args.name.strip()
    base_url = normalize_base_url(args.base_url)
    endpoints = [e for e in data.get("endpoints", []) if e.get("name") != name]
    endpoints.append({
        "name": name,
        "base_url": base_url,
        "enabled": not args.disabled,
        "tags": split_tags(args.tags),
        "notes": args.notes or "",
        "created_at_utc": utc_now_iso(),
        "updated_at_utc": utc_now_iso(),
    })
    data["endpoints"] = sorted(endpoints, key=lambda e: e.get("name", ""))
    save_registry(path, data)
    print(f"saved endpoint {name} -> {base_url} in {path}")
    return 0


def cmd_remove(args: argparse.Namespace) -> int:
    path = registry_path(args.registry)
    data = load_registry(path)
    before = len(data.get("endpoints", []))
    data["endpoints"] = [e for e in data.get("endpoints", []) if e.get("name") != args.name]
    save_registry(path, data)
    print(f"removed {before - len(data['endpoints'])} endpoint(s) from {path}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    path = registry_path(args.registry)
    data = load_registry(path)
    endpoints = select_endpoints(data, args.name, split_tags(args.tags), enabled_only=not args.all)
    if args.json:
        print(json.dumps({"registry": str(path), "endpoints": endpoints}, indent=2))
        return 0
    print(f"Registry: {path}")
    if not endpoints:
        print("No endpoints registered.")
        return 1
    for e in endpoints:
        status = "enabled" if e.get("enabled", True) else "disabled"
        print(f"- {e.get('name')}: {e.get('base_url')} [{status}] tags={','.join(e.get('tags') or [])}")
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    path = registry_path(args.registry)
    data = load_registry(path)
    endpoints = select_endpoints(data, args.name, split_tags(args.tags), enabled_only=not args.all)
    results = []
    for e in endpoints:
        probe = probe_endpoint(e["base_url"], timeout=args.timeout)
        results.append({**e, "probe": probe})
    if args.json:
        print(json.dumps({"registry": str(path), "results": results}, indent=2))
    else:
        for item in results:
            p = item["probe"]
            print(f"{item.get('name')}: {'OK' if p['reachable'] else 'FAIL'} {item.get('base_url')} models={p['model_count']} error={p['error'] or ''}")
    return 0 if results and all(r["probe"]["reachable"] for r in results) else 1


def cmd_export_inventory(args: argparse.Namespace) -> int:
    path = registry_path(args.registry)
    data = load_registry(path)
    endpoints = select_endpoints(data, args.name, split_tags(args.tags), enabled_only=not args.all)
    rows = []
    endpoint_id = 1
    model_id = 1
    for e in endpoints:
        p = probe_endpoint(e["base_url"], timeout=args.timeout)
        models = p["models"]
        if args.max_models > 0:
            models = models[: args.max_models]
        for model in models:
            rows.append({
                "host_name": e.get("name") or socket.gethostname(),
                "host_ip": args.host_ip or local_host_ip(),
                "endpoint_id": endpoint_id,
                "base_url": e["base_url"],
                "reachable": 1 if p["reachable"] else 0,
                "model_id": model_id,
                "model_key": model,
            })
            model_id += 1
        endpoint_id += 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["host_name", "host_ip", "endpoint_id", "base_url", "reachable", "model_id", "model_key"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out} with {len(rows)} model row(s)")
    return 0 if rows else 1


def cmd_set_enabled(args: argparse.Namespace, enabled: bool) -> int:
    path = registry_path(args.registry)
    data = load_registry(path)
    changed = 0
    for e in data.get("endpoints", []):
        if e.get("name") == args.name:
            e["enabled"] = enabled
            e["updated_at_utc"] = utc_now_iso()
            changed += 1
    save_registry(path, data)
    print(f"updated {changed} endpoint(s)")
    return 0 if changed else 1


def cmd_disable(args: argparse.Namespace) -> int:
    return cmd_set_enabled(args, False)


def cmd_enable(args: argparse.Namespace) -> int:
    return cmd_set_enabled(args, True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage persistent lms-bench endpoint registry.")
    parser.add_argument("--registry", default=None, help=f"Registry file; default {DEFAULT_REGISTRY}")
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add")
    add.add_argument("name")
    add.add_argument("base_url")
    add.add_argument("--tags", default=None)
    add.add_argument("--notes", default=None)
    add.add_argument("--disabled", action="store_true")
    add.set_defaults(func=cmd_add)

    remove = sub.add_parser("remove")
    remove.add_argument("name")
    remove.set_defaults(func=cmd_remove)

    enable = sub.add_parser("enable")
    enable.add_argument("name")
    enable.set_defaults(func=cmd_enable)

    disable = sub.add_parser("disable")
    disable.add_argument("name")
    disable.set_defaults(func=cmd_disable)

    list_cmd = sub.add_parser("list")
    list_cmd.add_argument("--name", action="append", default=[])
    list_cmd.add_argument("--tags", default=None)
    list_cmd.add_argument("--all", action="store_true")
    list_cmd.add_argument("--json", action="store_true")
    list_cmd.set_defaults(func=cmd_list)

    probe = sub.add_parser("probe")
    probe.add_argument("--name", action="append", default=[])
    probe.add_argument("--tags", default=None)
    probe.add_argument("--all", action="store_true")
    probe.add_argument("--timeout", type=int, default=8)
    probe.add_argument("--json", action="store_true")
    probe.set_defaults(func=cmd_probe)

    inv = sub.add_parser("export-inventory")
    inv.add_argument("--name", action="append", default=[])
    inv.add_argument("--tags", default=None)
    inv.add_argument("--all", action="store_true")
    inv.add_argument("--timeout", type=int, default=8)
    inv.add_argument("--out", default="lmstudio_inventory.csv")
    inv.add_argument("--max-models", type=int, default=0)
    inv.add_argument("--host-ip", default=None)
    inv.set_defaults(func=cmd_export_inventory)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
