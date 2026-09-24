#!/usr/bin/env python3
"""Fleet node reporter.

Phones the auto-router's pubsub fleet endpoint every --interval seconds with this
node's LM Studio library (the full downloaded catalog), its currently-loaded
models, and *real* machine specs. The orchestrator uses the published specs to
build per-node RAM budgets, so they must come from the node itself -- not from a
benchmark run stamped by whatever host executed bench_fleet.py.

Usage:
  python3 fleet_node_reporter.py --router-url http://100.64.43.123:8088 --interval 30
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from lms_agent_bench import runtime_identity_witness as _runtime_witness


LM_PORT = 1234


# --------------------------------------------------------------------------- #
# Specs (real hardware on the node running this reporter)
# --------------------------------------------------------------------------- #
def _read_proc_meminfo() -> Dict[str, float]:
    """Parse /proc/meminfo into gibibyte values for the keys we care about."""
    out: Dict[str, float] = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.split(":")
                if len(parts) != 2:
                    continue
                key = parts[0].strip()
                val = parts[1].strip().split()
                if len(val) < 2 or val[1] != "kB":
                    continue
                try:
                    out[key] = int(val[0]) / (1024 * 1024)
                except ValueError:
                    continue
    except OSError:
        pass
    return out


def _linux_cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as fh:
            seen = False
            for line in fh:
                if line.lower().startswith("model name"):
                    val = line.split(":", 1)[1].strip()
                    if val:
                        return val
                if line.lower().startswith("processor") and not seen:
                    seen = True
    except OSError:
        pass
    return ""


def _linux_gpu() -> str:
    """Best-effort GPU detection on Linux. Returns a short label or ''."""
    # nvidia-smi is the most reliable signal of a usable accelerator.
    if shutil.which("nvidia-smi"):
        try:
            txt = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=10,
            ).strip()
            names = [n.strip() for n in txt.splitlines() if n.strip()]
            if names:
                return ", ".join(sorted(set(names)))
        except (subprocess.SubprocessError, OSError):
            pass
    # Fall back to scanning lspci for known GPU vendors.
    if shutil.which("lspci"):
        try:
            txt = subprocess.check_output(
                ["lspci"], text=True, stderr=subprocess.DEVNULL, timeout=10
            )
            hits = []
            for line in txt.splitlines():
                low = line.lower()
                if "vga compatible controller" in low or "3d controller" in low:
                    if "nvidia" in low:
                        hits.append("NVIDIA")
                    elif "amd" in low or "advanced micro devices" in low:
                        hits.append("AMD")
                    elif "intel" in low:
                        hits.append("Intel")
            if hits:
                return ", ".join(sorted(set(hits)))
        except (subprocess.SubprocessError, OSError):
            pass
    return ""


def _macos_memory() -> Dict[str, float]:
    """Return {MemTotal_gib, MemAvailable_gib} on macOS using sysctl + vm_stat."""
    out: Dict[str, float] = {}
    try:
        mem_bytes = int(
            subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
        out["MemTotal_gib"] = mem_bytes / (1024**3)
    except (subprocess.SubprocessError, OSError, ValueError):
        return out
    # vm_stat reports page counts; default page size 4096 on Apple silicon/Intel.
    try:
        txt = subprocess.check_output(
            ["vm_stat"], text=True, stderr=subprocess.DEVNULL, timeout=10
        )
        pages = 4096
        free_pages = 0
        inactive_pages = 0
        for line in txt.splitlines():
            if line.startswith("Pages free:"):
                free_pages = int(line.split(":")[1].strip().replace(".", ""))
            elif line.startswith("Pages inactive:"):
                inactive_pages = int(line.split(":")[1].strip().replace(".", ""))
        # Treat free + inactive as reclaimable available memory (conservative).
        out["MemAvailable_gib"] = (free_pages + inactive_pages) * pages / (1024**3)
    except (subprocess.SubprocessError, OSError, ValueError):
        pass
    return out


def _macos_cpu_model() -> str:
    try:
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return ""


def _macos_gpu() -> str:
    if shutil.which("system_profiler"):
        try:
            txt = subprocess.check_output(
                ["system_profiler", "SPDisplaysDataType"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=15,
            )
            for line in txt.splitlines():
                s = line.strip()
                if s.startswith("Chipset Model:"):
                    return s.split(":", 1)[1].strip()
        except (subprocess.SubprocessError, OSError):
            pass
    return "Apple"  # macOS without discrete GPU is almost always Apple silicon/IGPU


def _specs() -> Dict[str, Any]:
    """Collect real, node-local hardware specs. Never falls back to benchmark
    artifacts -- if we cannot read a value we leave it blank rather than lie."""
    specs: Dict[str, Any] = {
        "platform": platform.system(),
        "hostname": socket.gethostname(),
        "cpu_model": "",
        "cpu_cores": os.cpu_count() or 0,
        "gpu": "",
        "system_ram_gib": 0.0,
        "available_ram_gib": 0.0,
    }
    if platform.system() == "Linux":
        mem = _read_proc_meminfo()
        specs["system_ram_gib"] = round(mem.get("MemTotal", 0.0), 2)
        specs["available_ram_gib"] = round(mem.get("MemAvailable", 0.0), 2)
        specs["cpu_model"] = _linux_cpu_model()
        specs["gpu"] = _linux_gpu()
    elif platform.system() == "Darwin":
        mem = _macos_memory()
        specs["system_ram_gib"] = round(mem.get("MemTotal_gib", 0.0), 2)
        specs["available_ram_gib"] = round(
            mem.get("MemAvailable_gib", mem.get("MemTotal_gib", 0.0)), 2
        )
        specs["cpu_model"] = _macos_cpu_model()
        specs["gpu"] = _macos_gpu()
    else:
        # Unknown platform: report what little we can, no fabrication.
        specs["cpu_model"] = platform.processor()
    return specs


# --------------------------------------------------------------------------- #
# LM Studio state
# --------------------------------------------------------------------------- #
def _http_get_json(url: str, timeout: float = 5.0) -> Optional[Any]:
    try:
        req = Request(url, headers={"Accept": "application/json"})
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:  # noqa: BLE001 - reporter must never crash the loop
        return None


def _loaded_models(lm_url: str) -> List[str]:
    """Currently-loaded model identifiers via the LM Studio native API.

    Use /api/v1/models and filter to items where loaded_instances is non-empty
    (actually resident in VRAM/RAM). The top-level 'loaded' boolean is unreliable.
    """
    data = _http_get_json(f"{lm_url.rstrip('/')}/api/v1/models")
    if not isinstance(data, dict):
        return []
    items = data.get("models") or []
    out = []
    for m in items:
        if not isinstance(m, dict):
            continue
        # LM Studio /api/v1/models uses 'key' field (not 'id')
        mid = m.get("key") or m.get("model_key") or m.get("path") or m.get("id")
        if not mid:
            continue
        # Only include models that are actually resident (loaded_instances non-empty)
        loaded_instances = m.get("loaded_instances") or []
        if isinstance(loaded_instances, list) and len(loaded_instances) > 0:
            out.append(mid)
    return out


def _library(lm_url: str) -> List[Dict[str, Any]]:
    """Full downloaded catalog (loaded or not) via the LM Studio native API."""
    data = _http_get_json(f"{lm_url.rstrip('/')}/api/v0/models")
    if not isinstance(data, dict):
        return []
    items = data.get("data") or []
    out = []
    for m in items:
        if not isinstance(m, dict):
            continue
        mid = m.get("id") or m.get("model_key") or m.get("path")
        if mid:
            out.append({"id": str(mid), "path": m.get("path", "")})
    return out


def _normalize_runtime_url(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if "://" not in raw:
        raw = "http://" + raw
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return ""
    path = parsed.path.rstrip("/")
    if path == "/v1":
        path = ""
    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            path,
            "",
            "",
            "",
        )
    ).rstrip("/")


def _configured_runtime_urls(
    lm_url: str,
    extra_urls: Optional[List[str]] = None,
) -> List[str]:
    candidates = [lm_url]
    candidates.extend(extra_urls or [])
    candidates.extend(
        item.strip()
        for item in os.getenv("FLEET_RUNTIME_URLS", "").split(",")
        if item.strip()
    )
    out: List[str] = []
    seen = set()
    for candidate in candidates:
        normalized = _normalize_runtime_url(candidate)
        if normalized and normalized not in seen:
            seen.add(normalized)
            out.append(normalized)
    return out



def _configured_runtime_witness_paths(
    extra_paths: Optional[List[str]] = None,
) -> List[str]:
    candidates = list(extra_paths or [])
    candidates.extend(
        item.strip()
        for item in os.getenv("FLEET_RUNTIME_WITNESSES", "").split(",")
        if item.strip()
    )
    out: List[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = str(Path(candidate).expanduser().resolve())
        if resolved not in seen:
            seen.add(resolved)
            out.append(resolved)
    return out


def _runtime_witnesses(
    extra_paths: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for raw_path in _configured_runtime_witness_paths(extra_paths):
        path = Path(raw_path)
        signature_path = Path(str(path) + ".sig")
        try:
            payload = path.read_bytes()
            if len(payload) > 16 * 1024:
                continue
            witness = json.loads(payload.decode("utf-8"))
            if not isinstance(witness, dict):
                continue
            if witness.get("schema_version") != _runtime_witness.SCHEMA_VERSION:
                continue
            if _runtime_witness._canonical_bytes(witness) != payload:  # noqa: SLF001
                continue
            signature = signature_path.read_text(encoding="utf-8")
            if len(signature) > 8 * 1024 or "BEGIN SSH SIGNATURE" not in signature:
                continue
            runtime_url = _normalize_runtime_url(str(witness.get("runtime_url") or ""))
            if not runtime_url or runtime_url in out:
                continue
            out[runtime_url] = {
                "witness": witness,
                "payload": payload.decode("utf-8"),
                "signature": signature,
            }
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            continue
    return out

def _openai_models(runtime_url: str) -> Optional[List[str]]:
    data = _http_get_json(f"{runtime_url.rstrip('/')}/v1/models")
    if not isinstance(data, dict):
        return None
    items = data.get("data")
    if not isinstance(items, list):
        return None
    return sorted(
        {
            str(item.get("id")).strip()
            for item in items
            if isinstance(item, dict) and str(item.get("id") or "").strip()
        }
    )


def _runtime_observation(
    runtime_url: str,
    hostname: str,
    witness_record: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Observe one serving process without granting it admission authority.

    This intentionally separates *discovery* from the approved AssistX runtime
    projection. A reachable process can become visible immediately, while
    artifact identity / capacity / access-path evidence still has to pass the
    existing operator-reviewed admission gates before Auto-Router may use it.
    """

    normalized = _normalize_runtime_url(runtime_url)
    if not normalized:
        return None

    native = _http_get_json(f"{normalized}/api/v1/models")
    if isinstance(native, dict) and isinstance(native.get("models"), list):
        runtime_kind = "lmstudio"
        models = sorted(set(_loaded_models(normalized)))
        protocol = "lmstudio-native"
    else:
        generic_models = _openai_models(normalized)
        if generic_models is None:
            return None
        runtime_kind = "openai_compatible"
        models = generic_models
        protocol = "openai-compatible"

    seed = f"{hostname}|{normalized}|{runtime_kind}"
    observation_id = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]
    observation = {
        "observation_schema": "fleet-runtime-observation.v1",
        "runtime_observation_id": f"runtime-observation:{observation_id}",
        "runtime_kind": runtime_kind,
        "protocol": protocol,
        "base_url": normalized,
        "models": models,
        # A parseable /models response with no loaded/served models is not a
        # ready serving runtime. Keep this evidence conservative: readiness is
        # observational only and never grants admission.
        "ready": bool(models),
        "observed_model_count": len(models),
        "observed_at": int(time.time()),
        # Observation is evidence only. Never allow the reporter to mint an
        # admitted RuntimeInstance by implication.
        "admitted": False,
    }
    if witness_record is not None:
        witness = witness_record.get("witness")
        if isinstance(witness, dict):
            observation["runtime_identity_witness_json"] = witness_record.get("payload")
            observation["runtime_identity_witness_signature"] = witness_record.get("signature")
            observation["runtime_identity_continuity"] = (
                _runtime_witness.observe_process_continuity(witness)
            )
    return observation


def _runtime_observations(
    lm_url: str,
    hostname: str,
    extra_urls: Optional[List[str]] = None,
    witness_paths: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    observations = []
    witnesses = _runtime_witnesses(witness_paths)
    for runtime_url in _configured_runtime_urls(lm_url, extra_urls):
        normalized = _normalize_runtime_url(runtime_url)
        observation = _runtime_observation(
            runtime_url,
            hostname,
            witnesses.get(normalized),
        )
        if observation is not None:
            observations.append(observation)
    return sorted(
        observations,
        key=lambda item: (str(item["runtime_kind"]), str(item["base_url"])),
    )


# --------------------------------------------------------------------------- #
# Report + loop
# --------------------------------------------------------------------------- #
def build_report(
    lm_url: str,
    runtime_urls: Optional[List[str]] = None,
    runtime_witnesses: Optional[List[str]] = None,
) -> Dict[str, Any]:
    specs = _specs()
    hostname = socket.gethostname()
    return {
        "hostname": hostname,
        "host_name": hostname,
        "library": _library(lm_url),
        "loaded": _loaded_models(lm_url),
        "runtimes": _runtime_observations(
            lm_url,
            hostname,
            runtime_urls,
            runtime_witnesses,
        ),
        "specs": specs,
    }


def post_report(router_url: str, report: Dict[str, Any], timeout: float = 10.0) -> bool:
    url = f"{router_url.rstrip('/')}/api/fleet/node-report"
    data = json.dumps(report).encode("utf-8")
    req = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except (URLError, HTTPError, OSError) as exc:
        print(f"post_report failed: {exc}", file=sys.stderr)
        return False


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Fleet node reporter")
    parser.add_argument("--router-url", default="http://100.64.43.123:8088")
    parser.add_argument("--lmstudio-url", default=f"http://localhost:{LM_PORT}")
    parser.add_argument(
        "--runtime-url",
        action="append",
        default=[],
        help=(
            "Additional local inference runtime root URL. Repeat for llama.cpp, "
            "SGLang, vLLM, secondary LM Studio servers, or other OpenAI-compatible "
            "runtimes. FLEET_RUNTIME_URLS may also provide a comma-separated list."
        ),
    )
    parser.add_argument(
        "--runtime-witness",
        action="append",
        default=[],
        help=(
            "Canonical signed runtime-identity witness JSON. Repeat per runtime; "
            "the detached OpenSSH signature must be beside it as <path>.sig. "
            "FLEET_RUNTIME_WITNESSES may also provide comma-separated paths."
        ),
    )
    parser.add_argument("--interval", type=float, default=30.0)
    args = parser.parse_args(argv)

    print(
        f"reporter: router={args.router_url} lmstudio={args.lmstudio_url} "
        f"extra_runtimes={len(args.runtime_url)} "
        f"runtime_witnesses={len(args.runtime_witness)} interval={args.interval}s",
        flush=True,
    )
    while True:
        report = build_report(
            args.lmstudio_url,
            args.runtime_url,
            args.runtime_witness,
        )
        ok = post_report(args.router_url, report)
        print(
            f"report {report['hostname']}: library={len(report['library'])} "
            f"loaded={len(report['loaded'])} runtimes={len(report['runtimes'])} "
            f"ram={report['specs'].get('system_ram_gib')} "
            f"avail={report['specs'].get('available_ram_gib')} posted={ok}",
            flush=True,
        )
        time.sleep(max(1.0, args.interval))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
