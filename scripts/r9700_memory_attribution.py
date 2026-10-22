#!/usr/bin/env python3
"""Read-only, fail-closed R9700 memory attribution report.

No process manipulation, swapping, memory pressure, GPU access, GPU load,
secrets, kernel writes, service restarts, or inference. Nested memory cgroup
values are hierarchical: report top-level subtraction only, never add every
child recursively. Attribution gaps are NOT automatically leaks.
"""
from __future__ import annotations

import argparse
import json
import re
import socket
from datetime import datetime, timezone
from pathlib import Path

_GIB = 1024**3
_MAX_CGROUP_DIRS = 2048
_MAX_CGROUP_DEPTH = 8
_MAX_PID_DIRS = 32768
_RAM_TARGET_GIB = 60
_AVAILABLE_GATE_GIB = 8
_CGROUP_THRESHOLD = 64 * 1024**2
_MISSING = "UNAVAILABLE"


def _read_bounded(path: Path, *, limit: int = 256_000) -> str:
    try:
        with path.open("rb") as stream:
            blob = stream.read(limit + 1)
    except (OSError, ValueError):
        return ""
    if len(blob) > limit:
        return ""
    try:
        return blob.decode("utf-8")
    except UnicodeDecodeError:
        return ""


def _kv_kib(text: str) -> dict[str, int]:
    result = {}
    for line in text.splitlines():
        match = re.fullmatch(r"([A-Za-z_()]+):\s*(\d+)\s*(?:kB)?", line.strip())
        if match:
            result[match[1]] = int(match[2])
    return result


def _kv_bytes(text: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].isdigit():
            result[parts[0]] = int(parts[1])
    return result


def _psi(text: str) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for line in text.splitlines():
        parts = line.split()
        if not parts or parts[0] not in ("some", "full"):
            continue
        try:
            fields = dict(p.split("=", 1) for p in parts[1:] if "=" in p)
            result[parts[0]] = {
                k: float(fields[k]) for k in ("avg10", "avg60", "avg300")
            }
        except (ValueError, KeyError):
            continue
    return result


def _cgroup_snapshot(cgroup_root: Path) -> dict:
    try:
        iterator = cgroup_root.rglob("memory.stat")
    except OSError:
        return {"root_available": False, "top_consumers": [], "top_level_unattributed_mib": None}
    observed: list[dict] = []
    for i, stat_path in enumerate(iterator):
        if i >= _MAX_CGROUP_DIRS:
            break
        parent = stat_path.parent
        relative = parent.relative_to(cgroup_root)
        if len(relative.parts) > _MAX_CGROUP_DEPTH:
            continue
        counters = _kv_bytes(_read_bounded(stat_path))
        if "shmem" not in counters:
            continue
        current_raw = _read_bounded(parent / "memory.current", limit=64).strip()
        current = int(current_raw) if current_raw.isdecimal() else None
        observed.append({
            "cgroup": relative.as_posix(),
            "shmem_bytes": counters["shmem"],
            "anon_bytes": counters.get("anon"),
            "memory_current_bytes": current,
        })
    root_entry = next((row for row in observed if row["cgroup"] == "."), None)
    first_level = [
        row for row in observed
        if row["cgroup"] != "." and "/" not in row["cgroup"]
    ]
    observed.sort(key=lambda row: row["shmem_bytes"], reverse=True)
    # Hierarchical values: do NOT add user.slice plus its nested children.
    unmatched = (
        max(0, root_entry["shmem_bytes"] - sum(x["shmem_bytes"] for x in first_level))
        if root_entry is not None else None
    )
    top = [
        {
            "cgroup": row["cgroup"],
            "shmem_mib": round(row["shmem_bytes"] / 2**20, 2),
            "anon_mib": round((row["anon_bytes"] or 0) / 2**20, 2),
            "charged_memory_mib": (
                round(row["memory_current_bytes"] / 2**20, 2)
                if row["memory_current_bytes"] is not None else None
            ),
        }
        for row in observed if row["shmem_bytes"] >= _CGROUP_THRESHOLD
    ][:20]
    return {
        "root_available": root_entry is not None,
        "top_consumers": top,
        "top_level_unattributed_mib": (
            round(unmatched / 2**20, 2) if unmatched is not None else None
        ),
        "unattributed_interpretation": (
            "Root aggregate minus VISIBLE top-level children only. Includes "
            "possible root-charged, unlisted or hierarchy/accounting effects; "
            "not a measured leak or a claim about any process."
        ),
        "sampled_cgroup_count": len(observed),
    }


def _llama_process_snapshot(proc_root: Path) -> list[dict]:
    result = []
    try:
        children = proc_root.iterdir()
    except OSError:
        return result
    examined = 0
    for p in children:
        if not p.name.isdecimal():
            continue
        examined += 1
        if examined > _MAX_PID_DIRS:
            break
        if _read_bounded(p / "comm", limit=128).strip() != "llama-server":
            continue
        kv = _kv_kib(_read_bounded(p / "status", limit=64000))
        smaps = _kv_kib(_read_bounded(p / "smaps_rollup", limit=64000))
        result.append({
            "pid": int(p.name),
            "vmrss_mib": round(kv["VmRSS"] / 1024, 2) if "VmRSS" in kv else None,
            "vmswap_mib": round(kv["VmSwap"] / 1024, 2) if "VmSwap" in kv else None,
            "pss_shmem_mib": (
                round(smaps["Pss_Shmem"] / 1024, 2)
                if "Pss_Shmem" in smaps else None
            ),
            "smaps_readable": "Pss_Shmem" in smaps,
        })
    return sorted(result, key=lambda row: row["pid"])


def collect(
    *, proc_root: Path = Path("/proc"), cgroup_root: Path = Path("/sys/fs/cgroup"),
    target_host: str | None = None,
) -> dict:
    target_host = socket.gethostname() if target_host is None else target_host
    if target_host != "joyner":
        return {
            "schema": "r9700-readonly-memory-attribution.v1",
            "host_observed": target_host,
            "status": "HOST_MISMATCH",
            "production_admission": "NO_GO",
            "model_launch_allowed": False,
        }
    mem = _kv_kib(_read_bounded(proc_root / "meminfo"))
    psi = _psi(_read_bounded(proc_root / "pressure" / "memory"))
    capacity = mem.get("MemTotal")
    available = mem.get("MemAvailable")
    required = {
        "MemTotal", "MemAvailable", "Shmem", "SwapTotal", "SwapFree"
    }
    complete = (
        required.issubset(mem)
        and all(mem[x] >= 0 for x in required)
        and available <= capacity
        and mem["SwapFree"] <= mem["SwapTotal"]
        and "some" in psi and "full" in psi
    )
    if not complete:
        return {
            "schema": "r9700-readonly-memory-attribution.v1",
            "status": "INCOMPLETE",
            "production_admission": "NO_GO",
            "model_launch_allowed": False,
            "error": "kernel memory or PSI data unavailable or inconsistent",
        }
    observed = {
        "installed_gib": round(capacity / 1024**2, 3),
        "available_gib": round(available / 1024**2, 3),
        "shmem_mib": round(mem["Shmem"] / 1024, 2),
        "swap_used_gib": round((mem["SwapTotal"]-mem["SwapFree"]) / 1024**2, 3),
        "swap_free_gib": round(mem["SwapFree"] / 1024**2, 3),
        "memory_psi": psi,
    }
    return {
        "schema": "r9700-readonly-memory-attribution.v1",
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "host_observed": target_host,
        "status": "READ_ONLY_OBSERVATION",
        "observed_memory": observed,
        "cgroup_shared_memory": _cgroup_snapshot(cgroup_root),
        "existing_llama_server_processes": _llama_process_snapshot(proc_root),
        "memory_capacity_gate": "HOLD" if capacity < _RAM_TARGET_GIB * _GIB else "CAPACITY_ONLY",
        "available_memory_gate": "HOLD" if available < _AVAILABLE_GATE_GIB * _GIB else "HEADROOM_ONLY",
        "cause_attribution": "NOT_PROVEN",
        "full_model_benchmark_executed": False,
        "gpu_loaded": False,
        "production_admission": "NO_GO",
        "model_launch_allowed": False,
        "safe_next_action": (
            "Correlate cgroup charges with approved operator evidence; obtain "
            "64 GiB RAM upgrade and verified >=8 GiB available headroom, "
            "then a separately approved GPU maintenance window. "
            "Do not clear swap, kill processes, or load IQ3_S."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proc-root", type=Path, default=Path("/proc"))
    parser.add_argument("--cgroup-root", type=Path, default=Path("/sys/fs/cgroup"))
    args = parser.parse_args(argv)
    result = collect(proc_root=args.proc_root, cgroup_root=args.cgroup_root)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result["status"] == "READ_ONLY_OBSERVATION" else 2


if __name__ == "__main__":
    raise SystemExit(main())