#!/usr/bin/env python3
"""Discover live local inference processes and run read-only artifact evidence probes.

This helper deliberately avoids shell PID splitting and never mutates runtime,
routing, projection, or admission state.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from lms_agent_bench import runtime_identity_witness as witness

_PROCESS_PATTERN = re.compile(
    r"(LM Studio|llmworker|lmlink|mlx|llama|qwen|ollama|vllm|sglang)",
    re.IGNORECASE,
)
_WEIGHT_SUFFIXES = (
    ".safetensors",
    ".gguf",
    ".bin",
    ".pt",
    ".pth",
)


def _run(args: list[str], *, timeout: int = 20) -> str:
    process = subprocess.run(
        args,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if process.returncode != 0:
        return ""
    return process.stdout


def candidate_processes() -> list[dict[str, Any]]:
    output = _run(["/bin/ps", "axo", "pid=,ppid=,command="])
    out: list[dict[str, Any]] = []
    for raw in output.splitlines():
        line = raw.strip()
        if not line or not _PROCESS_PATTERN.search(line):
            continue
        parts = line.split(maxsplit=2)
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
        except ValueError:
            continue
        out.append({"pid": pid, "ppid": ppid, "command": parts[2]})
    return out


def open_weight_paths(pid: int) -> list[Path]:
    output = _run(["/usr/sbin/lsof", "-nP", "-a", "-p", str(pid), "-Fn"])
    paths: set[Path] = set()
    for raw in output.splitlines():
        if not raw.startswith("n"):
            continue
        value = raw[1:]
        lower = value.lower()
        if not lower.endswith(_WEIGHT_SUFFIXES):
            continue
        path = Path(value)
        if path.is_file():
            paths.add(path)
    return sorted(paths, key=lambda item: str(item))


def vmmap_weight_paths(pid: int) -> list[str]:
    output = _run(["/usr/bin/vmmap", "-w", str(pid)], timeout=30)
    found: set[str] = set()
    for line in output.splitlines():
        lower = line.lower()
        if not any(suffix in lower for suffix in _WEIGHT_SUFFIXES):
            continue
        # vmmap lines contain address/range metadata before a path. Extract the
        # first absolute path through the final recognized weight suffix.
        match = re.search(
            r"(/[^\n]*?\.(?:safetensors|gguf|bin|pt|pth))(?:\s|$)",
            line,
            re.IGNORECASE,
        )
        if match:
            found.add(match.group(1))
    return sorted(found)


def inspect_process(pid: int) -> dict[str, Any]:
    open_paths = open_weight_paths(pid)
    mapped_paths = vmmap_weight_paths(pid)
    probes: list[dict[str, Any]] = []
    for path in open_paths:
        try:
            result = {
                "model_path": str(path),
                **witness.process_identity(pid),
            }
            binding = witness._process_references_model(  # noqa: SLF001
                pid,
                path,
                witness._stat_identity(path.stat()),  # noqa: SLF001
            )
            result["model_process_binding"] = binding
            result["strong_binding"] = binding in {
                "proc_maps",
                "darwin_vmmap_lsof",
            }
            result["status"] = "strong" if result["strong_binding"] else "diagnostic"
        except (OSError, ValueError) as exc:
            result = {
                "model_path": str(path),
                "status": "unverified",
                "strong_binding": False,
                "error": str(exc),
            }
        probes.append(result)
    return {
        "pid": pid,
        "open_weight_paths": [str(path) for path in open_paths],
        "vmmap_weight_paths": mapped_paths,
        "probes": probes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only live runtime/artifact discovery and evidence probe."
    )
    parser.add_argument(
        "--pid",
        type=int,
        action="append",
        default=[],
        help="Inspect only this PID. May be repeated.",
    )
    args = parser.parse_args()

    processes = candidate_processes()
    wanted = set(args.pid)
    if wanted:
        processes = [item for item in processes if item["pid"] in wanted]

    inspected = []
    for process in processes:
        evidence = inspect_process(int(process["pid"]))
        inspected.append({**process, **evidence})

    payload = {
        "schema_version": "fleet-runtime-discovery-evidence.v1",
        "platform": witness._platform_name(),  # noqa: SLF001
        "candidate_count": len(inspected),
        "weight_file_count": sum(
            len(item["open_weight_paths"]) for item in inspected
        ),
        "strong_probe_count": sum(
            1
            for item in inspected
            for probe in item["probes"]
            if probe.get("strong_binding") is True
        ),
        "mutating": False,
        "admission_authority": False,
        "processes": inspected,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
