#!/usr/bin/env python3
"""Collect REAL local host hardware (CPU/RAM/GPU) and emit JSON to stdout.

Intended to run ON each fleet node (piped over SSH) so the fleet writeup gets
per-node hardware instead of the runner's. Stdlib-only; platform-aware
(Linux /proc, macOS sysctl). No repo deps.

  ssh scott@node.tailnet 'python3 -s' < collect_node_profile.py
"""
from __future__ import annotations
import json, os, platform, socket, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path


def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def cpu_info():
    if platform.system() == "Darwin":
        return {"model": sh("sysctl -n machdep.cpu.brand_string") or platform.processor(),
                "physical_cores": int(sh("sysctl -n hw.physicalcpu") or 0) or None,
                "logical_processors": int(sh("sysctl -n hw.logicalcpu") or 0) or os.cpu_count(),
                "arch": platform.machine()}
    raw = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore") if Path("/proc/cpuinfo").exists() else ""
    model = None
    for b in [x for x in raw.split("\n\n") if x.strip()]:
        d = {}
        for line in b.splitlines():
            if ":" in line:
                k, v = line.split(":", 1); d[k.strip()] = v.strip()
        if "processor" in d and d.get("model name") and not model:
            model = d["model name"]
    return {"model": model or platform.processor(),
            "physical_cores": int(sh("nproc")) or os.cpu_count(),
            "logical_processors": os.cpu_count(),
            "arch": platform.machine()}


def mem_info():
    if platform.system() == "Darwin":
        total = int(sh("sysctl -n hw.memsize") or 0)
        pg = int(sh("sysctl -n hw.pagesize") or 4096)
        vs = sh("vm_stat")
        d = {k.strip(): int(v.strip().replace(".", "")) for k, v in
             (line.split(":", 1) for line in vs.splitlines() if ":" in line)}
        free = (d.get("Pages free", 0) + d.get("Pages inactive", 0)) * pg
        return {"ram_total_gib": round(total / 1073741824, 1),
                "ram_avail_gib": round(free / 1073741824, 1)}
    raw = Path("/proc/meminfo").read_text(encoding="utf-8", errors="ignore") if Path("/proc/meminfo").exists() else ""
    d = {}
    for line in raw.splitlines():
        if ":" in line:
            k, v = line.split(":", 1); d[k.strip()] = v.strip()
    total = int(d.get("MemTotal", "0").split()[0]) * 1024
    avail = int(d.get("MemAvailable", d.get("MemFree", "0")).split()[0]) * 1024
    return {"ram_total_gib": round(total / 1073741824, 1),
            "ram_avail_gib": round(avail / 1073741824, 1)}


def gpu_info():
    if platform.system() == "Darwin":
        out = sh("system_profiler SPDisplaysDataType 2>/dev/null | grep -E 'Chipset Model|Model'")
        return [l.split(":", 1)[-1].strip() for l in out.splitlines() if l.strip()][:4]
    try:
        out = subprocess.check_output(["lspci"], text=True, stderr=subprocess.DEVNULL)
        return [l.split(":", 2)[-1].strip() for l in out.splitlines()
                if any(k in l.lower() for k in ("vga", "3d", "display controller", "nvidia", "amd", "intel"))][:4]
    except Exception:
        return []


def vram_info():
    # NVIDIA
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total",
                                       "--format=csv,noheader,nounits"], text=True, stderr=subprocess.DEVNULL)
        vals = [int(x.strip()) for x in out.splitlines() if x.strip()]
        if vals:
            return {"vram_total_mib": sum(vals), "vendor": "nvidia"}
    except Exception:
        pass
    # AMD / ROCm
    try:
        out = subprocess.check_output(["rocm-smi", "--showmeminfo", "vram"],
                                      text=True, stderr=subprocess.DEVNULL)
        tot = 0
        for line in out.splitlines():
            if "Total" in line and "MiB" in line:
                tot += int(line.split(":")[-1].strip().split()[0])
        if tot:
            return {"vram_total_mib": tot, "vendor": "amd"}
    except Exception:
        pass
    return {}


def main():
    print(json.dumps({
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.system(),
        "cpu": cpu_info(),
        "memory": mem_info(),
        "gpu": gpu_info(),
        "vram": vram_info(),
        "source": "collect_node_profile.py (real per-node, run on host)",
    }, indent=2))


if __name__ == "__main__":
    main()
