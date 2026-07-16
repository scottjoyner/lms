#!/usr/bin/env python3
"""
Collect a local machine profile for LMS agent benchmark runs.

The profiler is intentionally dependency-light and Ubuntu-friendly. It uses Python
standard library APIs first, then optional shell commands when available. Missing
commands are recorded as warnings instead of failing the run.

Examples:
  python3 lms_machine_profile.py --output-dir runs/manual-profile
  python3 lms_machine_profile.py --inventory-csv lmstudio_inventory.csv --output-dir runs/manual-profile
  python3 lms_machine_profile.py --probe-base-url http://127.0.0.1:1234/v1 --output-dir runs/local
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import textwrap
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_TIMEOUT_S = 6


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def run_command(cmd: List[str], timeout_s: int = DEFAULT_TIMEOUT_S) -> Dict[str, Any]:
    """Run a command and return a structured result without raising."""
    if not shutil.which(cmd[0]):
        return {
            "command": cmd,
            "available": False,
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": f"command not found: {cmd[0]}",
        }

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        return {
            "command": cmd,
            "available": True,
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": cmd,
            "available": True,
            "ok": False,
            "returncode": None,
            "stdout": (exc.stdout or "").strip() if isinstance(exc.stdout, str) else "",
            "stderr": f"timeout after {timeout_s}s",
        }
    except Exception as exc:  # defensive: profile collection should never kill the run
        return {
            "command": cmd,
            "available": shutil.which(cmd[0]) is not None,
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": repr(exc),
        }


def parse_json_command(cmd: List[str], timeout_s: int = DEFAULT_TIMEOUT_S) -> Tuple[Optional[Any], Dict[str, Any]]:
    result = run_command(cmd, timeout_s=timeout_s)
    if not result["ok"] or not result["stdout"]:
        return None, result
    try:
        return json.loads(result["stdout"]), result
    except json.JSONDecodeError as exc:
        result = dict(result)
        result["json_error"] = str(exc)
        return None, result


def read_text(path: str) -> Optional[str]:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def parse_os_release() -> Dict[str, str]:
    raw = read_text("/etc/os-release") or ""
    parsed: Dict[str, str] = {}
    for line in raw.splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        parsed[key] = value.strip().strip('"')
    return parsed


def parse_meminfo() -> Dict[str, Any]:
    raw = read_text("/proc/meminfo") or ""
    values_kib: Dict[str, int] = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        match = re.search(r"(\d+)", rest)
        if match:
            values_kib[key] = int(match.group(1))

    def kib_to_bytes(key: str) -> Optional[int]:
        val = values_kib.get(key)
        return val * 1024 if val is not None else None

    total = kib_to_bytes("MemTotal")
    available = kib_to_bytes("MemAvailable")
    swap_total = kib_to_bytes("SwapTotal")
    swap_free = kib_to_bytes("SwapFree")
    return {
        "mem_total_bytes": total,
        "mem_available_bytes": available,
        "mem_used_estimate_bytes": (total - available) if total is not None and available is not None else None,
        "swap_total_bytes": swap_total,
        "swap_free_bytes": swap_free,
        "raw_kib": values_kib,
    }


def parse_cpuinfo() -> Dict[str, Any]:
    raw = read_text("/proc/cpuinfo") or ""
    processors = 0
    model_name = None
    cpu_mhz_values: List[float] = []
    flags: Optional[List[str]] = None
    for block in raw.split("\n\n"):
        if not block.strip():
            continue
        item: Dict[str, str] = {}
        for line in block.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                item[key.strip()] = value.strip()
        if "processor" in item:
            processors += 1
        if not model_name and item.get("model name"):
            model_name = item["model name"]
        if item.get("cpu MHz"):
            try:
                cpu_mhz_values.append(float(item["cpu MHz"]))
            except ValueError:
                pass
        if flags is None and item.get("flags"):
            flags = item["flags"].split()

    return {
        "model_name": model_name or platform.processor() or None,
        "logical_processors": processors or os.cpu_count(),
        "cpu_mhz_min_observed": min(cpu_mhz_values) if cpu_mhz_values else None,
        "cpu_mhz_max_observed": max(cpu_mhz_values) if cpu_mhz_values else None,
        "flags": flags or [],
    }


def collect_cpu() -> Dict[str, Any]:
    lscpu_json, lscpu_result = parse_json_command(["lscpu", "--json"])
    cpuinfo = parse_cpuinfo()
    parsed: Dict[str, Any] = {
        "source": "proc_cpuinfo",
        **cpuinfo,
        "lscpu": None,
        "lscpu_command": summarize_command_result(lscpu_result),
    }
    if isinstance(lscpu_json, dict):
        parsed["source"] = "lscpu+proc_cpuinfo"
        parsed["lscpu"] = lscpu_json
        # Pull common values into top-level fields for easy reporting.
        fields = {
            item.get("field", "").strip(":"): item.get("data")
            for item in lscpu_json.get("lscpu", [])
            if isinstance(item, dict)
        }
        parsed.update(
            {
                "architecture": fields.get("Architecture"),
                "cpu_op_modes": fields.get("CPU op-mode(s)"),
                "vendor_id": fields.get("Vendor ID"),
                "model_name": fields.get("Model name") or parsed.get("model_name"),
                "sockets": fields.get("Socket(s)"),
                "cores_per_socket": fields.get("Core(s) per socket"),
                "threads_per_core": fields.get("Thread(s) per core"),
                "logical_processors": int(fields.get("CPU(s)", parsed.get("logical_processors") or 0)) or parsed.get("logical_processors"),
            }
        )
    return parsed


def summarize_command_result(result: Dict[str, Any], max_chars: int = 2000) -> Dict[str, Any]:
    return {
        "command": result.get("command"),
        "available": result.get("available"),
        "ok": result.get("ok"),
        "returncode": result.get("returncode"),
        "stderr": (result.get("stderr") or "")[:max_chars],
        "json_error": result.get("json_error"),
    }


def collect_storage() -> Dict[str, Any]:
    root_usage = shutil.disk_usage("/")
    mounts: List[Dict[str, Any]] = []
    mountinfo = read_text("/proc/mounts") or ""
    seen_mounts = set()
    for line in mountinfo.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        device, mountpoint, fstype = parts[:3]
        if mountpoint in seen_mounts:
            continue
        seen_mounts.add(mountpoint)
        if not mountpoint.startswith(("/",)):
            continue
        try:
            usage = shutil.disk_usage(mountpoint)
        except Exception:
            continue
        # Avoid huge noise from pseudo filesystems.
        if fstype in {"proc", "sysfs", "devtmpfs", "devpts", "securityfs", "cgroup", "cgroup2", "pstore", "bpf", "tracefs", "debugfs", "mqueue", "hugetlbfs"}:
            continue
        mounts.append(
            {
                "device": device,
                "mountpoint": mountpoint,
                "fstype": fstype,
                "total_bytes": usage.total,
                "used_bytes": usage.used,
                "free_bytes": usage.free,
            }
        )

    lsblk_json, lsblk_result = parse_json_command(["lsblk", "--json", "--bytes", "-o", "NAME,TYPE,SIZE,MODEL,SERIAL,MOUNTPOINTS,FSTYPE"], timeout_s=8)
    return {
        "root_total_bytes": root_usage.total,
        "root_used_bytes": root_usage.used,
        "root_free_bytes": root_usage.free,
        "mounts": mounts,
        "lsblk": lsblk_json,
        "lsblk_command": summarize_command_result(lsblk_result),
    }


def parse_lspci_gpus(stdout: str) -> List[Dict[str, str]]:
    gpus: List[Dict[str, str]] = []
    for line in stdout.splitlines():
        lower = line.lower()
        if any(token in lower for token in ["vga compatible controller", "3d controller", "display controller"]):
            gpus.append({"lspci_line": line})
    return gpus


def collect_nvidia() -> Dict[str, Any]:
    query = "index,name,driver_version,memory.total,memory.used,memory.free,temperature.gpu,power.draw,power.limit,utilization.gpu"
    result = run_command(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"], timeout_s=8)
    devices: List[Dict[str, Any]] = []
    if result["ok"] and result["stdout"]:
        for line in result["stdout"].splitlines():
            values = [x.strip() for x in line.split(",")]
            if len(values) != len(query.split(",")):
                continue
            item = dict(zip(query.split(","), values))
            devices.append(item)
    return {"available": bool(result.get("available")), "ok": bool(result.get("ok")), "devices": devices, "command": summarize_command_result(result)}


def collect_rocm() -> Dict[str, Any]:
    # rocm-smi JSON output varies across versions. Preserve raw parsed data when available.
    rocm_json, result = parse_json_command(["rocm-smi", "--showall", "--json"], timeout_s=10)
    return {"available": bool(result.get("available")), "ok": bool(result.get("ok")), "data": rocm_json, "command": summarize_command_result(result)}


def collect_gpu() -> Dict[str, Any]:
    lspci = run_command(["lspci"], timeout_s=8)
    vulkan = run_command(["vulkaninfo", "--summary"], timeout_s=8)
    vainfo = run_command(["vainfo"], timeout_s=8)
    return {
        "lspci_gpus": parse_lspci_gpus(lspci.get("stdout", "")) if lspci.get("ok") else [],
        "lspci_command": summarize_command_result(lspci),
        "nvidia": collect_nvidia(),
        "rocm": collect_rocm(),
        "vulkan_summary": vulkan.get("stdout", "")[:6000] if vulkan.get("ok") else "",
        "vulkan_command": summarize_command_result(vulkan),
        "vainfo_summary": vainfo.get("stdout", "")[:6000] if vainfo.get("ok") else "",
        "vainfo_command": summarize_command_result(vainfo),
    }


def normalize_base_url(url: str) -> str:
    url = url.strip().rstrip("/")
    if not url:
        return url
    if url.endswith("/v1"):
        return url
    return url + "/v1"


def probe_lmstudio_base_url(base_url: str, timeout_s: int = DEFAULT_TIMEOUT_S) -> Dict[str, Any]:
    base_url = normalize_base_url(base_url)
    models_url = base_url.rstrip("/") + "/models"
    started = dt.datetime.now(dt.timezone.utc)
    try:
        req = urllib.request.Request(models_url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read(2_000_000).decode("utf-8", errors="replace")
            elapsed = (dt.datetime.now(dt.timezone.utc) - started).total_seconds()
            parsed = json.loads(body)
            models = parsed.get("data", []) if isinstance(parsed, dict) else []
            return {
                "base_url": base_url,
                "models_url": models_url,
                "reachable": True,
                "status": getattr(resp, "status", None),
                "elapsed_s": elapsed,
                "model_count": len(models) if isinstance(models, list) else None,
                "models": [m.get("id") for m in models if isinstance(m, dict) and m.get("id")],
                "error": None,
            }
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        elapsed = (dt.datetime.now(dt.timezone.utc) - started).total_seconds()
        return {
            "base_url": base_url,
            "models_url": models_url,
            "reachable": False,
            "status": None,
            "elapsed_s": elapsed,
            "model_count": None,
            "models": [],
            "error": repr(exc),
        }


def load_inventory_endpoints(path: Optional[str]) -> List[Dict[str, Any]]:
    if not path:
        return []
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("base_url"):
                rows.append(row)
    return rows


def collect_lmstudio(args: argparse.Namespace) -> Dict[str, Any]:
    base_urls = set()
    inventory_rows = load_inventory_endpoints(args.inventory_csv)
    for row in inventory_rows:
        if row.get("base_url"):
            base_urls.add(normalize_base_url(row["base_url"]))
    for url in args.probe_base_url or []:
        base_urls.add(normalize_base_url(url))

    probes = [probe_lmstudio_base_url(url, timeout_s=args.timeout) for url in sorted(base_urls)]
    return {
        "inventory_csv": args.inventory_csv,
        "inventory_row_count": len(inventory_rows),
        "unique_endpoint_count": len(base_urls),
        "endpoint_probes": probes,
    }


def bytes_to_gib(value: Optional[int]) -> Optional[float]:
    if value is None:
        return None
    return round(value / (1024 ** 3), 2)


def derive_recommendations(profile: Dict[str, Any]) -> List[str]:
    recs: List[str] = []
    mem_total = profile.get("memory", {}).get("mem_total_bytes")
    mem_gib = bytes_to_gib(mem_total)
    gpu = profile.get("gpu", {})
    nvidia_devices = gpu.get("nvidia", {}).get("devices", []) or []
    rocm_ok = gpu.get("rocm", {}).get("ok")
    lspci_gpus = gpu.get("lspci_gpus", []) or []
    endpoints = profile.get("lmstudio", {}).get("endpoint_probes", []) or []
    reachable_endpoints = [e for e in endpoints if e.get("reachable")]

    if mem_gib is not None:
        if mem_gib < 16:
            recs.append("System RAM is below 16 GiB; prefer small quantized models and short-context tasks.")
        elif mem_gib < 32:
            recs.append("System RAM is moderate; use 7B/8B quantized models for routine tasks and avoid large long-context runs.")
        elif mem_gib < 64:
            recs.append("System RAM can support stronger local models, but long context should still be benchmarked before routing agent work.")
        else:
            recs.append("System RAM is suitable for heavier local model testing and multi-model benchmark sweeps.")

    if nvidia_devices:
        recs.append("NVIDIA GPU runtime detected; prioritize GPU-loaded models and record VRAM fit per model.")
    elif rocm_ok:
        recs.append("ROCm runtime detected; benchmark AMD GPU acceleration carefully because model/backend support varies.")
    elif lspci_gpus:
        recs.append("GPU hardware is visible, but no NVIDIA/ROCm runtime was confirmed; expect CPU or limited acceleration unless LM Studio reports otherwise.")
    else:
        recs.append("No GPU was detected through lspci; route only small/CPU-friendly models to this host until verified.")

    if endpoints:
        if reachable_endpoints:
            recs.append(f"{len(reachable_endpoints)} LM Studio endpoint(s) were reachable during profiling; benchmark these first.")
        else:
            recs.append("LM Studio endpoint rows were present but none were reachable; verify LM Studio server mode, bind address, firewall, and Tailscale routing.")
    else:
        recs.append("No LM Studio endpoints were provided to the profiler; run with --probe-base-url or --inventory-csv before benchmarking.")

    return recs


def collect_profile(args: argparse.Namespace) -> Dict[str, Any]:
    os_release = parse_os_release()
    profile: Dict[str, Any] = {
        "schema_version": "machine_profile.v1",
        "created_at_utc": utc_now_iso(),
        "host": {
            "hostname": socket.gethostname(),
            "fqdn": socket.getfqdn(),
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "python_version": sys.version.split()[0],
        },
        "os_release": os_release,
        "cpu": collect_cpu(),
        "memory": parse_meminfo(),
        "storage": collect_storage(),
        "gpu": collect_gpu(),
        "lmstudio": collect_lmstudio(args),
        "warnings": [],
    }
    profile["recommendations"] = derive_recommendations(profile)
    profile["warnings"] = collect_warnings(profile)
    return profile


def collect_warnings(profile: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []
    for section, key in [
        ("cpu", "lscpu_command"),
        ("storage", "lsblk_command"),
        ("gpu", "lspci_command"),
    ]:
        cmd = profile.get(section, {}).get(key, {})
        if cmd and not cmd.get("ok"):
            warnings.append(f"{section}.{key} did not complete: {cmd.get('stderr')}")

    gpu = profile.get("gpu", {})
    for runtime in ["nvidia", "rocm"]:
        cmd = gpu.get(runtime, {}).get("command", {})
        if cmd and not cmd.get("ok"):
            warnings.append(f"{runtime} runtime probe unavailable or failed: {cmd.get('stderr')}")
    return warnings


def render_markdown(profile: Dict[str, Any]) -> str:
    host = profile.get("host", {})
    cpu = profile.get("cpu", {})
    mem = profile.get("memory", {})
    storage = profile.get("storage", {})
    gpu = profile.get("gpu", {})
    lmstudio = profile.get("lmstudio", {})

    lines: List[str] = []
    lines.append("# LMS Machine Profile")
    lines.append("")
    lines.append(f"- Generated UTC: `{profile.get('created_at_utc')}`")
    lines.append(f"- Hostname: `{host.get('hostname')}`")
    lines.append(f"- Platform: `{host.get('platform')}`")
    lines.append(f"- Python: `{host.get('python_version')}`")
    lines.append("")

    lines.append("## CPU")
    lines.append("")
    lines.append(f"- Model: `{cpu.get('model_name')}`")
    lines.append(f"- Architecture: `{cpu.get('architecture')}`")
    lines.append(f"- Logical processors: `{cpu.get('logical_processors')}`")
    lines.append(f"- Cores/socket: `{cpu.get('cores_per_socket')}`")
    lines.append(f"- Threads/core: `{cpu.get('threads_per_core')}`")
    lines.append("")

    lines.append("## Memory")
    lines.append("")
    lines.append(f"- Total RAM: `{bytes_to_gib(mem.get('mem_total_bytes'))} GiB`")
    lines.append(f"- Available RAM: `{bytes_to_gib(mem.get('mem_available_bytes'))} GiB`")
    lines.append(f"- Swap total: `{bytes_to_gib(mem.get('swap_total_bytes'))} GiB`")
    lines.append("")

    lines.append("## Storage")
    lines.append("")
    lines.append(f"- Root total: `{bytes_to_gib(storage.get('root_total_bytes'))} GiB`")
    lines.append(f"- Root free: `{bytes_to_gib(storage.get('root_free_bytes'))} GiB`")
    lines.append("")

    lines.append("## GPU / acceleration")
    lines.append("")
    lspci_gpus = gpu.get("lspci_gpus", []) or []
    if lspci_gpus:
        for item in lspci_gpus:
            lines.append(f"- `{item.get('lspci_line')}`")
    else:
        lines.append("- No GPU detected by `lspci`, or `lspci` unavailable.")
    nvidia_devices = gpu.get("nvidia", {}).get("devices", []) or []
    if nvidia_devices:
        lines.append("")
        lines.append("### NVIDIA")
        for dev in nvidia_devices:
            lines.append(f"- GPU {dev.get('index')}: `{dev.get('name')}`, VRAM `{dev.get('memory.total')} MiB`, driver `{dev.get('driver_version')}`")
    lines.append("")

    lines.append("## LM Studio endpoints")
    lines.append("")
    probes = lmstudio.get("endpoint_probes", []) or []
    if probes:
        lines.append("| Base URL | Reachable | Models | Latency s | Error |")
        lines.append("|---|:---:|---:|---:|---|")
        for p in probes:
            lines.append(
                f"| `{p.get('base_url')}` | {'yes' if p.get('reachable') else 'no'} | "
                f"{p.get('model_count') if p.get('model_count') is not None else ''} | "
                f"{round(p.get('elapsed_s') or 0, 3)} | `{(p.get('error') or '')[:120]}` |"
            )
    else:
        lines.append("No endpoints were provided. Use `--probe-base-url http://host:1234/v1` or `--inventory-csv`.")
    lines.append("")

    lines.append("## Recommendations")
    lines.append("")
    for rec in profile.get("recommendations", []):
        lines.append(f"- {rec}")
    lines.append("")

    warnings = profile.get("warnings", []) or []
    if warnings:
        lines.append("## Warnings")
        lines.append("")
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")

    return "\n".join(lines)


def write_outputs(profile: Dict[str, Any], args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = Path(args.json_out) if args.json_out else output_dir / "machine_profile.json"
    md_path = Path(args.md_out) if args.md_out else output_dir / "machine_synopsis.md"

    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(render_markdown(profile), encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect machine, hardware, and LM Studio endpoint profile for LMS benchmark runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Examples:
              python3 lms_machine_profile.py --output-dir runs/profile-local
              python3 lms_machine_profile.py --probe-base-url http://127.0.0.1:1234/v1 --output-dir runs/profile-local
              python3 lms_machine_profile.py --inventory-csv lmstudio_inventory.csv --output-dir runs/profile-from-inventory
            """
        ),
    )
    parser.add_argument("--output-dir", default="runs/machine_profile", help="Directory for machine_profile.json and machine_synopsis.md")
    parser.add_argument("--json-out", default=None, help="Optional explicit JSON output path")
    parser.add_argument("--md-out", default=None, help="Optional explicit Markdown output path")
    parser.add_argument("--inventory-csv", default=None, help="Optional LM Studio inventory CSV with base_url column")
    parser.add_argument("--probe-base-url", action="append", default=[], help="LM Studio OpenAI-compatible base URL to probe; may be repeated")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S, help="Per-command / endpoint probe timeout in seconds")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    profile = collect_profile(args)
    write_outputs(profile, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
