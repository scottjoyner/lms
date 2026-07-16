#!/usr/bin/env python3
"""Bridge to the official LM Studio `lms` CLI.

This project intentionally should not install a console command named `lms`,
because `lms` is LM Studio's own CLI. This module calls the official CLI when it
is present and converts useful outputs into benchmark inventory artifacts.

Useful official commands this wrapper understands:
  - lms --help
  - lms server status
  - lms server start --port N --bind 127.0.0.1
  - lms ls --json [--host HOST]
  - lms ps --json [--host HOST]
  - lms load --estimate-only MODEL [--context-length N] [--gpu max|auto|off]
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import socket
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_LMS_BIN = "lms"
DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"


def normalize_host(host: str) -> str:
    host = host.strip()
    if host.startswith("http://") or host.startswith("https://"):
        host = host.split("//", 1)[1]
    if host.endswith("/v1"):
        host = host[:-3]
    return host.rstrip("/")


def base_url_from_host(host: str, port: Optional[int] = None) -> str:
    host = normalize_host(host)
    if ":" not in host and port:
        host = f"{host}:{port}"
    return f"http://{host}/v1"


def local_host_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def run_lms(args: List[str], *, lms_bin: str = DEFAULT_LMS_BIN, timeout: int = 60) -> Dict[str, Any]:
    path = shutil.which(lms_bin)
    if not path:
        return {"ok": False, "available": False, "cmd": [lms_bin] + args, "returncode": None, "stdout": "", "stderr": f"{lms_bin} not found on PATH"}
    try:
        proc = subprocess.run([path] + args, capture_output=True, text=True, timeout=timeout, check=False)
        return {"ok": proc.returncode == 0, "available": True, "cmd": [path] + args, "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
    except subprocess.TimeoutExpired as exc:
        return {"ok": False, "available": True, "cmd": [path] + args, "returncode": None, "stdout": exc.stdout or "", "stderr": f"timeout after {timeout}s"}


def parse_json_stdout(result: Dict[str, Any]) -> Any:
    if not result.get("ok"):
        return None
    try:
        return json.loads(result.get("stdout") or "")
    except Exception:
        return None


def cmd_status(args: argparse.Namespace) -> int:
    result = run_lms(["server", "status"], lms_bin=args.lms_bin, timeout=args.timeout)
    print(json.dumps(result, indent=2 if args.json else None))
    return 0 if result.get("ok") else 1


def cmd_start_server(args: argparse.Namespace) -> int:
    cmd = ["server", "start"]
    if args.port:
        cmd += ["--port", str(args.port)]
    if args.bind:
        cmd += ["--bind", args.bind]
    if args.cors:
        cmd.append("--cors")
    result = run_lms(cmd, lms_bin=args.lms_bin, timeout=args.timeout)
    print(result.get("stdout") or result.get("stderr"))
    return 0 if result.get("ok") else 1


def extract_model_ids(data: Any) -> List[str]:
    ids: List[str] = []
    if isinstance(data, list):
        candidates = data
    elif isinstance(data, dict):
        candidates = data.get("models") or data.get("data") or data.get("llms") or []
    else:
        candidates = []
    if isinstance(candidates, dict):
        candidates = list(candidates.values())
    for item in candidates:
        if isinstance(item, str):
            ids.append(item)
        elif isinstance(item, dict):
            for key in ["modelKey", "model_key", "key", "id", "identifier", "path"]:
                if item.get(key):
                    ids.append(str(item[key]))
                    break
    return sorted(set(ids))


def list_models(host: Optional[str], loaded_only: bool, lms_bin: str, timeout: int) -> Dict[str, Any]:
    cmd = ["ps" if loaded_only else "ls", "--json"]
    if host:
        cmd += ["--host", normalize_host(host)]
    result = run_lms(cmd, lms_bin=lms_bin, timeout=timeout)
    data = parse_json_stdout(result)
    return {"result": result, "data": data, "models": extract_model_ids(data)}


def cmd_models(args: argparse.Namespace) -> int:
    payload = list_models(args.host, args.loaded_only, args.lms_bin, args.timeout)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"official lms available: {payload['result'].get('available')}")
        print(f"command ok: {payload['result'].get('ok')}")
        if payload["result"].get("stderr"):
            print(payload["result"]["stderr"].strip())
        for model in payload["models"]:
            print(model)
    return 0 if payload["result"].get("ok") else 1


def cmd_inventory(args: argparse.Namespace) -> int:
    payload = list_models(args.host, args.loaded_only, args.lms_bin, args.timeout)
    models = payload["models"]
    if args.max_models and args.max_models > 0:
        models = models[: args.max_models]
    host = args.host or f"127.0.0.1:{args.port}"
    base_url = args.base_url or base_url_from_host(host, None)
    rows = []
    for idx, model in enumerate(models, start=1):
        rows.append({
            "host_name": args.host_name or socket.gethostname(),
            "host_ip": args.host_ip or local_host_ip(),
            "endpoint_id": 1,
            "base_url": base_url,
            "reachable": 1 if payload["result"].get("ok") else 0,
            "model_id": idx,
            "model_key": model,
        })
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["host_name", "host_ip", "endpoint_id", "base_url", "reachable", "model_id", "model_key"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out} with {len(rows)} model row(s)")
    if not payload["result"].get("ok"):
        print(payload["result"].get("stderr") or payload["result"].get("stdout"))
    return 0 if rows else 1


def cmd_estimate(args: argparse.Namespace) -> int:
    cmd = ["load", "--estimate-only", args.model]
    if args.context_length:
        cmd += ["--context-length", str(args.context_length)]
    if args.gpu:
        cmd += ["--gpu", args.gpu]
    if args.host:
        cmd += ["--host", normalize_host(args.host)]
    result = run_lms(cmd, lms_bin=args.lms_bin, timeout=args.timeout)
    print(result.get("stdout") or result.get("stderr"))
    return 0 if result.get("ok") else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bridge to official LM Studio lms CLI without shadowing it.")
    parser.add_argument("--lms-bin", default=DEFAULT_LMS_BIN, help="Official LM Studio CLI binary name/path")
    parser.add_argument("--timeout", type=int, default=60)
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="Run official `lms server status`")
    status.add_argument("--json", action="store_true")
    status.set_defaults(func=cmd_status)

    start = sub.add_parser("start-server", help="Run official `lms server start`")
    start.add_argument("--port", type=int, default=None)
    start.add_argument("--bind", default=None)
    start.add_argument("--cors", action="store_true")
    start.set_defaults(func=cmd_start_server)

    models = sub.add_parser("models", help="List models through official lms ls/ps")
    models.add_argument("--host", default=None)
    models.add_argument("--loaded-only", action="store_true", help="Use `lms ps --json` instead of `lms ls --json`")
    models.add_argument("--json", action="store_true")
    models.set_defaults(func=cmd_models)

    inv = sub.add_parser("inventory", help="Create LMS benchmark inventory from official lms ls/ps")
    inv.add_argument("--host", default=None, help="Remote LM Studio host for official lms --host")
    inv.add_argument("--port", type=int, default=1234)
    inv.add_argument("--base-url", default=None, help="Explicit OpenAI-compatible base URL for benchmark HTTP calls")
    inv.add_argument("--out", default="lmstudio_inventory.csv")
    inv.add_argument("--loaded-only", action="store_true")
    inv.add_argument("--max-models", type=int, default=0)
    inv.add_argument("--host-name", default=None)
    inv.add_argument("--host-ip", default=None)
    inv.set_defaults(func=cmd_inventory)

    estimate = sub.add_parser("estimate", help="Run official `lms load --estimate-only MODEL`")
    estimate.add_argument("model")
    estimate.add_argument("--host", default=None)
    estimate.add_argument("--context-length", type=int, default=None)
    estimate.add_argument("--gpu", default=None)
    estimate.set_defaults(func=cmd_estimate)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
