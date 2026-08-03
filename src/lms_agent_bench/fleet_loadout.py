#!/usr/bin/env python3
"""Hardware discovery and evidence-based loadout planning for LMS fleets.

The module deliberately separates four states:

* observation: what a machine reports now;
* plan: loadouts that are safe to benchmark;
* evidence: benchmark results tied to one candidate ID;
* selection: desired state that is still *not admitted* until external gates pass.

It never deploys, restarts, loads, unloads, or registers a runtime.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SCHEMA_VERSION = "fleet_loadout.v1"
DEFAULT_CONTEXTS = (4096, 8192, 16384, 32768)
DEFAULT_TIMEOUT_S = 8


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_command(cmd: Sequence[str], timeout_s: int = DEFAULT_TIMEOUT_S) -> Dict[str, Any]:
    if not cmd or not shutil.which(str(cmd[0])):
        return {
            "command": list(cmd),
            "available": False,
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": f"command not found: {cmd[0] if cmd else '<empty>'}",
        }
    try:
        proc = subprocess.run(
            list(cmd), capture_output=True, text=True, timeout=timeout_s, check=False
        )
        return {
            "command": list(cmd),
            "available": True,
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except subprocess.TimeoutExpired:
        return {
            "command": list(cmd),
            "available": True,
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": f"timeout after {timeout_s}s",
        }
    except Exception as exc:
        return {
            "command": list(cmd),
            "available": True,
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": repr(exc),
        }


def command_json(cmd: Sequence[str], timeout_s: int = DEFAULT_TIMEOUT_S) -> Tuple[Any, Dict[str, Any]]:
    result = run_command(cmd, timeout_s=timeout_s)
    if not result["ok"] or not result["stdout"]:
        return None, result
    try:
        return json.loads(result["stdout"]), result
    except json.JSONDecodeError as exc:
        result = dict(result)
        result["json_error"] = str(exc)
        return None, result


def read_text(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def parse_meminfo() -> Dict[str, Any]:
    values: Dict[str, int] = {}
    for line in read_text("/proc/meminfo").splitlines():
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        match = re.search(r"(\d+)", rest)
        if match:
            values[key] = int(match.group(1)) * 1024
    return {
        "total_bytes": values.get("MemTotal"),
        "available_bytes": values.get("MemAvailable"),
        "swap_total_bytes": values.get("SwapTotal"),
        "source": "/proc/meminfo",
    }


def sysctl_value(name: str) -> Optional[str]:
    result = run_command(["sysctl", "-n", name])
    return result["stdout"] if result["ok"] and result["stdout"] else None


def int_or_none(value: Any) -> Optional[int]:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def collect_linux() -> Dict[str, Any]:
    lscpu, lscpu_cmd = command_json(["lscpu", "--json"])
    fields: Dict[str, str] = {}
    if isinstance(lscpu, dict):
        fields = {
            str(item.get("field", "")).rstrip(":"): str(item.get("data", ""))
            for item in lscpu.get("lscpu", [])
            if isinstance(item, dict)
        }
    pci = run_command(["lspci", "-nnk"])
    pci_text = pci["stdout"] if pci["ok"] else ""
    gpu_lines = [
        line.strip()
        for line in pci_text.splitlines()
        if any(token in line.lower() for token in ("vga compatible", "3d controller", "display controller"))
    ]
    npu_lines = [
        line.strip()
        for line in pci_text.splitlines()
        if any(token in line.lower() for token in ("xdna", "neural", "npu", "ai engine"))
    ]

    nvidia = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    nvidia_devices: List[Dict[str, Any]] = []
    if nvidia["ok"]:
        for line in nvidia["stdout"].splitlines():
            parts = [item.strip() for item in line.split(",")]
            if len(parts) == 5:
                nvidia_devices.append(
                    {
                        "index": int_or_none(parts[0]),
                        "name": parts[1],
                        "driver_version": parts[2],
                        "memory_total_mib": int_or_none(parts[3]),
                        "memory_free_mib": int_or_none(parts[4]),
                    }
                )

    vulkan = run_command(["vulkaninfo", "--summary"], timeout_s=12)
    rocm = run_command(["rocminfo"], timeout_s=12)
    xrt_json, xrt_cmd = command_json(["xrt-smi", "examine", "--format", "JSON"], timeout_s=12)
    accel_devices = sorted(glob.glob("/dev/accel/accel*"))
    render_devices = sorted(glob.glob("/dev/dri/renderD*"))

    backends = ["cpu"]
    if nvidia_devices:
        backends.insert(0, "cuda")
    if rocm["ok"]:
        backends.insert(0, "rocm")
    if vulkan["ok"] or gpu_lines:
        backends.insert(0, "vulkan")
    if accel_devices or xrt_cmd["ok"] or npu_lines:
        backends.insert(0, "npu_xdna2")

    return {
        "cpu": {
            "model": fields.get("Model name") or fields.get("Model"),
            "architecture": fields.get("Architecture") or platform.machine(),
            "logical_processors": int_or_none(fields.get("CPU(s)")) or os.cpu_count(),
            "cores_per_socket": int_or_none(fields.get("Core(s) per socket")),
            "threads_per_core": int_or_none(fields.get("Thread(s) per core")),
            "flags": fields.get("Flags", "").split(),
            "raw_lscpu": lscpu,
            "probe": lscpu_cmd,
        },
        "memory": parse_meminfo(),
        "accelerators": {
            "gpu_pci": gpu_lines,
            "npu_pci": npu_lines,
            "nvidia": nvidia_devices,
            "vulkan": {
                "available": vulkan["ok"],
                "summary": vulkan["stdout"][:12000] if vulkan["ok"] else "",
                "probe": vulkan,
            },
            "rocm": {"available": rocm["ok"], "summary": rocm["stdout"][:12000], "probe": rocm},
            "xdna": {
                "available": bool(accel_devices or xrt_cmd["ok"] or npu_lines),
                "accel_devices": accel_devices,
                "render_devices": render_devices,
                "xrt": xrt_json,
                "probe": xrt_cmd,
            },
        },
        "supported_backends": list(dict.fromkeys(backends)),
    }


def collect_macos() -> Dict[str, Any]:
    profiler, profiler_cmd = command_json(
        ["system_profiler", "SPHardwareDataType", "SPDisplaysDataType", "-json"], timeout_s=20
    )
    mem_total = int_or_none(sysctl_value("hw.memsize"))
    cpu_model = sysctl_value("machdep.cpu.brand_string") or sysctl_value("hw.model")
    logical = int_or_none(sysctl_value("hw.logicalcpu"))
    physical = int_or_none(sysctl_value("hw.physicalcpu"))
    gpu_items: List[Dict[str, Any]] = []
    if isinstance(profiler, dict):
        for item in profiler.get("SPDisplaysDataType", []) or []:
            if isinstance(item, dict):
                gpu_items.append(item)
    return {
        "cpu": {
            "model": cpu_model,
            "architecture": platform.machine(),
            "logical_processors": logical or os.cpu_count(),
            "physical_processors": physical,
        },
        "memory": {
            "total_bytes": mem_total,
            "available_bytes": None,
            "swap_total_bytes": None,
            "source": "sysctl hw.memsize",
            "unified": True,
        },
        "accelerators": {
            "metal": {"available": True, "system_profiler_displays": gpu_items},
            "system_profiler": profiler,
            "probe": profiler_cmd,
        },
        "supported_backends": ["metal", "cpu"],
    }


def normalize_base_url(url: str) -> str:
    value = url.strip().rstrip("/")
    return value if value.endswith("/v1") else value + "/v1"


def http_json(url: str, timeout_s: int = DEFAULT_TIMEOUT_S) -> Tuple[Any, Optional[str], Optional[int]]:
    try:
        request = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read(5_000_000).decode("utf-8", errors="replace")), None, getattr(response, "status", None)
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return None, repr(exc), None


def probe_endpoint(base_url: str, timeout_s: int = DEFAULT_TIMEOUT_S) -> Dict[str, Any]:
    base_url = normalize_base_url(base_url)
    models, model_error, model_status = http_json(base_url + "/models", timeout_s)
    capabilities, capability_error, capability_status = http_json(base_url + "/capabilities", timeout_s)
    model_ids: List[str] = []
    if isinstance(models, dict) and isinstance(models.get("data"), list):
        model_ids = [str(item.get("id")) for item in models["data"] if isinstance(item, dict) and item.get("id")]
    return {
        "base_url": base_url,
        "reachable": model_error is None,
        "models": model_ids,
        "model_status": model_status,
        "model_error": model_error,
        "capabilities": capabilities if isinstance(capabilities, dict) else None,
        "capability_status": capability_status,
        "capability_error": capability_error,
    }


def discover(endpoints: Sequence[str] = ()) -> Dict[str, Any]:
    system = platform.system()
    if system == "Darwin":
        hardware = collect_macos()
    elif system == "Linux":
        hardware = collect_linux()
    else:
        hardware = {
            "cpu": {"model": platform.processor(), "architecture": platform.machine(), "logical_processors": os.cpu_count()},
            "memory": {"total_bytes": None, "available_bytes": None},
            "accelerators": {},
            "supported_backends": ["cpu"],
        }
    identity = {
        "hostname": socket.gethostname(),
        "fqdn": socket.getfqdn(),
        "system": system,
        "release": platform.release(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
    }
    fingerprint_input = {"identity": identity, "hardware": hardware}
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "machine_observation",
        "observed_at_utc": utc_now_iso(),
        "identity": identity,
        "hardware": hardware,
        "endpoint_observations": [probe_endpoint(url) for url in endpoints],
        "observation_fingerprint": canonical_hash(fingerprint_input),
        "authority": {
            "kind": "observation",
            "may_admit_runtime": False,
            "may_modify_desired_state": False,
        },
    }


def load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_contexts(value: str) -> List[int]:
    contexts = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not contexts or any(item <= 0 for item in contexts):
        raise ValueError("contexts must contain positive integers")
    return contexts


def model_records(raw: Any) -> List[Dict[str, Any]]:
    records = raw.get("models", []) if isinstance(raw, dict) else raw
    if not isinstance(records, list):
        raise ValueError("model file must be a list or an object with a models list")
    normalized: List[Dict[str, Any]] = []
    for item in records:
        if not isinstance(item, dict) or not item.get("id"):
            raise ValueError("every model requires an id")
        record = dict(item)
        path = record.get("path")
        if not record.get("size_bytes") and path and Path(path).exists():
            record["size_bytes"] = Path(path).stat().st_size
        record.setdefault("format", "gguf")
        record.setdefault("max_context", 8192)
        normalized.append(record)
    return normalized


def backend_memory_budget(observation: Mapping[str, Any], backend: str) -> Optional[int]:
    hardware = observation.get("hardware", {})
    memory = hardware.get("memory", {})
    total = int_or_none(memory.get("total_bytes"))
    accelerators = hardware.get("accelerators", {})
    if backend == "cuda":
        values = [int_or_none(item.get("memory_free_mib")) for item in accelerators.get("nvidia", []) if isinstance(item, dict)]
        values = [value for value in values if value]
        return max(values) * 1024 * 1024 if values else None
    if total:
        reserve = max(4 * 1024**3, int(total * 0.20))
        return max(0, total - reserve)
    return None


def kv_bytes_per_token(model: Mapping[str, Any]) -> int:
    explicit = int_or_none(model.get("kv_bytes_per_token"))
    if explicit and explicit > 0:
        return explicit
    params = float(model.get("parameter_billions") or 1.0)
    return max(64 * 1024, int(params * 64 * 1024))


def estimated_memory_bytes(model: Mapping[str, Any], context: int, slots: int, backend: str) -> Optional[int]:
    size = int_or_none(model.get("size_bytes"))
    if size is None:
        return None
    workspace_ratio = 0.12 if backend in {"cuda", "rocm", "vulkan", "metal"} else 0.08
    workspace = int(size * workspace_ratio)
    kv = kv_bytes_per_token(model) * context * slots
    return size + workspace + kv


def thread_choices(observation: Mapping[str, Any]) -> List[int]:
    cpu = observation.get("hardware", {}).get("cpu", {})
    logical = int_or_none(cpu.get("logical_processors")) or 1
    physical = int_or_none(cpu.get("physical_processors"))
    if physical is None:
        cores_per_socket = int_or_none(cpu.get("cores_per_socket"))
        physical = cores_per_socket or max(1, logical // 2)
    return sorted({max(1, physical), max(1, logical)})


def candidate_id(candidate: Mapping[str, Any]) -> str:
    stable = {key: value for key, value in candidate.items() if key not in {"candidate_id", "benchmark_port"}}
    return canonical_hash(stable).split(":", 1)[1][:16]


def build_plan(
    observation: Mapping[str, Any],
    models: Sequence[Mapping[str, Any]],
    contexts: Sequence[int] = DEFAULT_CONTEXTS,
    max_candidates: int = 96,
) -> Dict[str, Any]:
    backends = list(observation.get("hardware", {}).get("supported_backends", []) or ["cpu"])
    threads = thread_choices(observation)
    candidates: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    next_port = 18080

    for model in models:
        max_context = int_or_none(model.get("max_context")) or max(contexts)
        model_contexts = [ctx for ctx in contexts if ctx <= max_context] or [min(contexts)]
        for backend in backends:
            budget = backend_memory_budget(observation, backend)
            if backend == "npu_xdna2":
                candidate = {
                    "engine": "npu-inference-server",
                    "backend": backend,
                    "model": dict(model),
                    "context_tokens": max(model_contexts),
                    "parallel_slots": 1,
                    "streaming_required": True,
                    "launch": {"mode": "existing_or_adapter", "bind_host": "127.0.0.1"},
                    "estimated_memory_bytes": None,
                    "memory_budget_bytes": budget,
                    "benchmark_port": next_port,
                }
                candidate["candidate_id"] = candidate_id(candidate)
                candidates.append(candidate)
                next_port += 1
                continue

            gpu_layer_choices = [0] if backend == "cpu" else [999, 0]
            flash_choices = [False] if backend == "cpu" else [True, False]
            slot_choices = [1, 2]
            for context in model_contexts:
                for slots in slot_choices:
                    estimate = estimated_memory_bytes(model, context, slots, backend)
                    if estimate is not None and budget is not None and estimate > budget:
                        rejected.append(
                            {
                                "model_id": model.get("id"),
                                "backend": backend,
                                "context_tokens": context,
                                "parallel_slots": slots,
                                "reason": "estimated_memory_exceeds_budget",
                                "estimated_memory_bytes": estimate,
                                "memory_budget_bytes": budget,
                            }
                        )
                        continue
                    for gpu_layers in gpu_layer_choices:
                        for flash_attention in flash_choices:
                            for thread_count in threads:
                                candidate = {
                                    "engine": "llama.cpp",
                                    "backend": backend,
                                    "model": dict(model),
                                    "context_tokens": context,
                                    "parallel_slots": slots,
                                    "threads": thread_count,
                                    "gpu_layers": gpu_layers,
                                    "flash_attention": flash_attention,
                                    "batch_size": 512 if backend != "cpu" else 256,
                                    "ubatch_size": 128,
                                    "estimated_memory_bytes": estimate,
                                    "memory_budget_bytes": budget,
                                    "benchmark_port": next_port,
                                    "launch": {
                                        "mode": "ephemeral_loopback_only",
                                        "bind_host": "127.0.0.1",
                                        "required_binary": "llama-server",
                                    },
                                }
                                candidate["candidate_id"] = candidate_id(candidate)
                                candidates.append(candidate)
                                next_port += 1
                                if len(candidates) >= max_candidates:
                                    break
                            if len(candidates) >= max_candidates:
                                break
                        if len(candidates) >= max_candidates:
                            break
                    if len(candidates) >= max_candidates:
                        break
                if len(candidates) >= max_candidates:
                    break
            if len(candidates) >= max_candidates:
                break
        if len(candidates) >= max_candidates:
            break

    plan_core = {
        "observation_fingerprint": observation.get("observation_fingerprint"),
        "candidates": candidates,
        "rejected_candidates": rejected,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "benchmark_plan",
        "created_at_utc": utc_now_iso(),
        **plan_core,
        "plan_fingerprint": canonical_hash(plan_core),
        "benchmark_contract": {
            "required_metrics": [
                "ok_rate",
                "eval_score_avg",
                "tps_med",
                "ttft_med",
                "memory_peak_bytes",
                "memory_headroom_ratio",
                "concurrency_ok",
                "crash_count",
            ],
            "required_gates": [
                "completion_canary",
                "streaming_canary",
                "concurrency_limit",
                "cancellation_behavior",
                "memory_headroom",
                "sustained_stability",
            ],
            "deployment_allowed": False,
            "bind_policy": "loopback_only",
        },
    }


def bool_value(value: Any, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass", "passed"}


def float_value(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_results_csv(path: str) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def select_loadout(plan: Mapping[str, Any], results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_id = {str(item.get("candidate_id")): item for item in plan.get("candidates", []) if isinstance(item, dict)}
    valid_rows = [row for row in results if str(row.get("candidate_id") or row.get("loadout_id")) in by_id]
    best_tps = max([float_value(row.get("tps_med")) for row in valid_rows] or [1.0])
    ranked: List[Dict[str, Any]] = []

    for row in valid_rows:
        cid = str(row.get("candidate_id") or row.get("loadout_id"))
        candidate = by_id[cid]
        ok_rate = float_value(row.get("ok_rate"))
        quality = max(float_value(row.get("eval_score_avg")), float_value(row.get("eval_ok_rate")))
        tps = float_value(row.get("tps_med"))
        ttft = float_value(row.get("ttft_med"), 999.0)
        headroom = float_value(row.get("memory_headroom_ratio"))
        concurrency_ok = bool_value(row.get("concurrency_ok"))
        streaming_ok = bool_value(row.get("streaming_ok"), default=True)
        cancellation_ok = bool_value(row.get("cancellation_ok"), default=False)
        crash_count = int(float_value(row.get("crash_count")))
        hard_failures: List[str] = []
        if ok_rate < 0.98:
            hard_failures.append("ok_rate_below_0.98")
        if crash_count > 0:
            hard_failures.append("crash_observed")
        if not concurrency_ok:
            hard_failures.append("concurrency_gate_failed")
        if not streaming_ok:
            hard_failures.append("streaming_gate_failed")
        if headroom < 0.10:
            hard_failures.append("memory_headroom_below_10_percent")

        throughput_score = min(1.0, tps / best_tps) if best_tps > 0 else 0.0
        latency_score = 1.0 / (1.0 + max(0.0, ttft))
        stability_score = 1.0 if concurrency_ok and crash_count == 0 else 0.0
        score = (
            ok_rate * 0.35
            + quality * 0.25
            + throughput_score * 0.15
            + latency_score * 0.10
            + min(max(headroom, 0.0), 1.0) * 0.10
            + stability_score * 0.05
        )
        ranked.append(
            {
                "candidate_id": cid,
                "score": round(score, 6),
                "eligible": not hard_failures,
                "hard_failures": hard_failures,
                "candidate": candidate,
                "metrics": dict(row),
                "gates": {
                    "completion": ok_rate >= 0.98,
                    "streaming": streaming_ok,
                    "concurrency": concurrency_ok,
                    "cancellation": cancellation_ok,
                    "memory_headroom": headroom >= 0.10,
                    "sustained_stability": crash_count == 0,
                },
            }
        )

    ranked.sort(key=lambda item: (item["eligible"], item["score"]), reverse=True)
    selected = next((item for item in ranked if item["eligible"]), None)
    fallback = next(
        (
            item
            for item in ranked
            if item["eligible"] and selected and item["candidate_id"] != selected["candidate_id"]
        ),
        None,
    )
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "selected_loadout",
        "created_at_utc": utc_now_iso(),
        "observation_fingerprint": plan.get("observation_fingerprint"),
        "plan_fingerprint": plan.get("plan_fingerprint"),
        "selected": selected,
        "fallback": fallback,
        "ranked_results": ranked,
        "admission": {
            "admitted": False,
            "reason": "selection is desired state only; live authority must verify path, identity, capacity, and freshness",
            "required_external_gates": [
                "physical_runtime_identity",
                "model_artifact_fingerprint",
                "container_path_reachability",
                "lan_preference_and_tailscale_fallback",
                "shared_slot_admission",
                "rollback_canary",
            ],
        },
    }
    artifact["selection_fingerprint"] = canonical_hash(
        {key: value for key, value in artifact.items() if key not in {"created_at_utc", "selection_fingerprint"}}
    )
    return artifact


def split_ips(value: str) -> List[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def normalize_tailscale_csv(path: str, include_all: bool = False) -> Dict[str, Any]:
    ignored_names = ("iphone", "raspberrypi", "macbook-pro")
    nodes: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            name = (row.get("Device name") or "").strip()
            os_name = (row.get("OS") or "").strip().lower()
            if not name:
                continue
            relevant_os = os_name in {"linux", "macos"}
            ignored = any(token in name.lower() for token in ignored_names)
            if not include_all and (not relevant_os or ignored):
                continue
            nodes.append(
                {
                    "node_id": name,
                    "os": row.get("OS"),
                    "os_version": row.get("OS Version"),
                    "domain": row.get("Domain"),
                    "tailscale_version": row.get("Tailscale version"),
                    "created": row.get("Created"),
                    "last_seen": row.get("Last seen"),
                    "key_expiry": row.get("Key expiry"),
                    "tailscale_ips": split_ips(row.get("Tailscale IPs") or ""),
                    "tailscale_ssh": bool_value(row.get("Tailscale SSH")),
                    "funnel": bool_value(row.get("Funnel")),
                    "focus": relevant_os and not ignored,
                }
            )
    nodes.sort(key=lambda item: item["node_id"])
    core = {"nodes": nodes, "source": Path(path).name}
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "tailscale_inventory",
        "created_at_utc": utc_now_iso(),
        **core,
        "inventory_fingerprint": canonical_hash(core),
        "redaction": {
            "device_ids_removed": True,
            "creator_emails_removed": True,
            "managed_by_removed": True,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Discover, plan, and select evidence-based fleet inference loadouts.")
    sub = parser.add_subparsers(dest="command", required=True)

    discover_cmd = sub.add_parser("discover", help="Collect a local Linux/macOS hardware observation")
    discover_cmd.add_argument("--endpoint", action="append", default=[])
    discover_cmd.add_argument("--out", default="machine_observation.json")

    plan_cmd = sub.add_parser("plan", help="Generate safe benchmark candidates from an observation and model inventory")
    plan_cmd.add_argument("--observation", required=True)
    plan_cmd.add_argument("--models", required=True)
    plan_cmd.add_argument("--contexts", default=",".join(str(item) for item in DEFAULT_CONTEXTS))
    plan_cmd.add_argument("--max-candidates", type=int, default=96)
    plan_cmd.add_argument("--out", default="benchmark_plan.json")

    select_cmd = sub.add_parser("select", help="Rank candidate benchmark rows and emit desired state")
    select_cmd.add_argument("--plan", required=True)
    select_cmd.add_argument("--results-csv", required=True)
    select_cmd.add_argument("--out", default="selected_loadout.json")

    inventory_cmd = sub.add_parser("inventory", help="Normalize a Tailscale admin CSV without account/device identifiers")
    inventory_cmd.add_argument("--tailscale-csv", required=True)
    inventory_cmd.add_argument("--include-all", action="store_true")
    inventory_cmd.add_argument("--out", default="tailscale_inventory.json")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "discover":
        artifact = discover(args.endpoint)
    elif args.command == "plan":
        artifact = build_plan(
            load_json(args.observation),
            model_records(load_json(args.models)),
            contexts=parse_contexts(args.contexts),
            max_candidates=args.max_candidates,
        )
    elif args.command == "select":
        artifact = select_loadout(load_json(args.plan), read_results_csv(args.results_csv))
    elif args.command == "inventory":
        artifact = normalize_tailscale_csv(args.tailscale_csv, include_all=args.include_all)
    else:
        raise AssertionError(args.command)
    write_json(args.out, artifact)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
