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
import http.client
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


LM_PORT = 1234

# Explicitly pinned Joyner loopback lanes. Discovery is opt-in, library-only.
# This does not authorize inference, measure GPU residency, or select routes.
_JOYNER_LANES = (
    (1234, "joyner-toolcall", "/home/scott/fleet-gpu/models/MiniCPM5-1B-Q4_K_M.gguf"),
    (1236, "joyner-ling", "/home/scott/fleet-gpu/models/Ling-3.0-tiny-abliterated-APEX-I-Compact.gguf"),
)


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


# --------------------------------------------------------------------------- #
# Opt-in Joyner llama.cpp catalog discovery (no model loaded/ready claims).
# --------------------------------------------------------------------------- #
def _joyner_loopback_json(port: int, endpoint: str) -> Optional[Dict[str, Any]]:
    """GET-only bounded local API response; no supplied URL, credentials or GPU work."""
    if port not in (1234, 1236) or endpoint not in ("/health", "/props", "/v1/models"):
        return None
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2.0)
    try:
        connection.request("GET", endpoint, headers={"Accept": "application/json"})
        response = connection.getresponse()
        if response.status != 200:  # also rejects redirects; never follow arbitrary URLs
            return None
        payload = response.read(256 * 1024 + 1)
        if len(payload) > 256 * 1024:
            return None
        value = json.loads(payload)
        return value if isinstance(value, dict) else None
    except (OSError, ValueError, UnicodeError, http.client.HTTPException):
        return None
    finally:
        connection.close()


def _joyner_llama_library() -> List[Dict[str, Any]]:
    """Require three matching local observations, report catalog only, never loaded."""
    if socket.gethostname() != "joyner":
        return []
    found: List[Dict[str, Any]] = []
    for port, alias, model_path in _JOYNER_LANES:
        health = _joyner_loopback_json(port, "/health")
        props = _joyner_loopback_json(port, "/props")
        catalog = _joyner_loopback_json(port, "/v1/models")
        if not (health and health.get("status") == "ok"):
            continue
        if not (props and props.get("model_alias") == alias
                and props.get("model_path") == model_path):
            continue
        data = catalog.get("data") if catalog else None
        if not isinstance(data, list) or not any(
            isinstance(item, dict) and item.get("id") == alias
            for item in data
        ):
            continue
        # Empty model path avoids emitting home paths via fleet data.
        found.append({"id": alias, "path": ""})
    return found


# --------------------------------------------------------------------------- #
# Report + loop
# --------------------------------------------------------------------------- #
def build_report(lm_url: str, *, joyner_llama_discovery: bool = False) -> Dict[str, Any]:
    specs = _specs()
    hostname = socket.gethostname()
    library = _library(lm_url)
    if joyner_llama_discovery and hostname == "joyner":
        existing = {row["id"] for row in library if isinstance(row, dict) and "id" in row}
        for lane in _joyner_llama_library():
            if lane["id"] not in existing:
                library.append(lane)
                existing.add(lane["id"])
    # Deliberately do not add llama.cpp HTTP health to "loaded"; the only
    # existing loaded-model detector is LM Studio-specific.
    return {
        "hostname": hostname,
        "host_name": hostname,
        "library": library,
        "loaded": _loaded_models(lm_url),
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
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument(
        "--joyner-llama-discovery", action="store_true",
        help="Opt-in, pinned loopback llama.cpp library entries on Joyner only; never loaded",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print one local report and exit without POST or service changes",
    )
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps(build_report(
            args.lmstudio_url, joyner_llama_discovery=args.joyner_llama_discovery,
        ), indent=2, sort_keys=True))
        return 0

    print(
        f"reporter: router={args.router_url} lmstudio={args.lmstudio_url} "
        f"interval={args.interval}s",
        flush=True,
    )
    while True:
        report = build_report(
            args.lmstudio_url, joyner_llama_discovery=args.joyner_llama_discovery,
        )
        ok = post_report(args.router_url, report)
        print(
            f"report {report['hostname']}: library={len(report['library'])} "
            f"loaded={len(report['loaded'])} ram={report['specs'].get('system_ram_gib')} "
            f"avail={report['specs'].get('available_ram_gib')} posted={ok}",
            flush=True,
        )
        time.sleep(max(1.0, args.interval))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())