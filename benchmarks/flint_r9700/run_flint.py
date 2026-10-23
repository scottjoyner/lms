#!/usr/bin/env python3
"""Isolated, no-service-change R9700 Flint GGUF performance qualification.

Only a 100%-SHA256-verified pinned model; a separate pinned llama.cpp build.
Never changes existing model processes/routes, installs hardware, or publishes GO.
Capture both successful and denied/aborted invocations with resource witnesses.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import tempfile
import subprocess
import time
import uuid

PIN = {
    "schema": "lms-flint-r9700-run.v1",
    "host": "joyner",
    "device_id": "0x7551",
    "device": "Vulkan1",
    "runtime_source_sha": "970b51ce7c9ba5fff0726c5e9994f1c2f0e4eedc",
    "hf_repo": "unsloth/flint-GGUF",
    "hf_revision": "20468edfc38ab43d6914cc88b08cbecb05d0fc62",
    "quant": "UD-IQ3_XXS",
    "filename": "flint-UD-IQ3_XXS.gguf",
    "bytes": 14535427520,
    "sha256": "7a0a095dde1d931063a5d571e19497d997edbc783eb1ed6b86529bfab9d94804",
}
STAGE = Path("/home/scott/experiments/flint-r9700-20261022")
GPU = Path("/sys/class/drm/card0/device")
GiB = 1024 ** 3
MIN_AVAIL_BYTES = int(3.4 * GiB)
ABORT_AVAIL_BYTES = int(2.0 * GiB)
MIN_VRAM_MARGIN = 2 * GiB
MAX_FULL_PSI = 1.0

# Source revision and SHA256 were independently verified from HF LFS metadata
# and the staged x1-370 full files. Never compare speeds across unlabeled quants.
QUANTS = {
    "IQ3_XXS": {"quant": "UD-IQ3_XXS", "filename": "flint-UD-IQ3_XXS.gguf",
                "bytes": 14535427520,
                "sha256": "7a0a095dde1d931063a5d571e19497d997edbc783eb1ed6b86529bfab9d94804"},
    "IQ2_XXS": {"quant": "UD-IQ2_XXS", "filename": "flint-UD-IQ2_XXS.gguf",
                "bytes": 11855541696,
                "sha256": "c1fa4a6d5ec67fbeff460c68829984e4aa3eea95dc9d2067691f19e5406deeee"},
}
PROFILES = {
    "smoke": {"n_prompt": "64", "n_gen": "16", "r": 1, "max_sec": 100},
    "balanced": {"n_prompt": "128,512", "n_gen": "32,128", "r": 2, "max_sec": 240},
    "throughput": {"n_prompt": "128,1024", "n_gen": "128", "r": 3, "max_sec": 300},
}


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def memory_snapshot() -> dict:
    mem = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, val = line.partition(":")
        if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            pieces = val.split()
            if len(pieces) < 2 or pieces[1] != "kB":
                raise ValueError("bad meminfo")
            mem[key] = int(pieces[0]) * 1024
    if set(mem) != {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
        raise ValueError("missing kernel memory counters")
    full = next((v for v in Path("/proc/pressure/memory").read_text().splitlines()
                 if v.startswith("full ")), None)
    if full is None:
        raise ValueError("missing PSI")
    fields = dict(v.split("=", 1) for v in full.split()[1:] if "=" in v)
    psi = float(fields.get("avg10", "-1"))
    if not 0 <= psi <= 100:
        raise ValueError("invalid PSI")
    if (min(mem.values()) < 0 or mem["MemAvailable"] > mem["MemTotal"] or
        mem["SwapFree"] > mem["SwapTotal"]):
        raise ValueError("inconsistent host memory")
    return {
        "available_gib": round(mem["MemAvailable"] / GiB, 3),
        "swap_free_mib": round(mem["SwapFree"] / (1024 ** 2), 2),
        "full_psi_10_percent": psi,
        "available_bytes": mem["MemAvailable"],
    }


def gpu_snapshot() -> dict:
    identity = (GPU / "device").read_text().strip()
    total = int((GPU / "mem_info_vram_total").read_text())
    used = int((GPU / "mem_info_vram_used").read_text())
    power = (GPU / "power/runtime_status").read_text().strip()
    if identity != PIN["device_id"] or not (0 <= used <= total):
        raise ValueError("R9700 PCI/VRAM identity mismatch")
    return {"pci_device": identity, "total_gib": round(total / GiB, 3),
            "used_gib": round(used / GiB, 3),
            "free_bytes": total - used, "runtime_power": power}


def pinned_regular(path: Path, expected: int | None = None) -> os.stat_result:
    if not path.is_absolute() or path.is_symlink():
        raise ValueError("unsafe or relative pinned path")
    s = path.lstat()
    if not stat.S_ISREG(s.st_mode):
        raise ValueError("nonregular input")
    if expected is not None and s.st_size != expected:
        raise ValueError("wrong pinned model byte count")
    return s


def hash_pinned(path: Path, expected_size: int, expected_sha256: str) -> str:
    first = pinned_regular(path, expected_size)
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    with os.fdopen(fd, "rb") as reader:
        if os.fstat(reader.fileno()).st_ino != first.st_ino:
            raise ValueError("model inode changed before read")
        cursor = 0
        released = 0
        for buf in iter(lambda: reader.read(4 * 1024 ** 2), b""):
            digest.update(buf)
            cursor += len(buf)
            # This newly staged GGUF is not used by production lanes.
            # Hint to the kernel to release *our* read-only hash page cache,
            # rather than crowding Joyner's 16GiB host. Advisory only.
            if hasattr(os, "posix_fadvise") and cursor - released >= 64 * 1024 ** 2:
                try:
                    os.posix_fadvise(reader.fileno(), released, cursor - released,
                                    os.POSIX_FADV_DONTNEED)
                except OSError:
                    pass
                released = cursor
        after = os.fstat(reader.fileno())
    last = pinned_regular(path, expected_size)
    if (first.st_dev, first.st_ino, first.st_mtime_ns, first.st_ctime_ns) != (
        after.st_dev, after.st_ino, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError("model changed while being hashed")
    if (last.st_dev, last.st_ino, last.st_mtime_ns, last.st_ctime_ns) != (
        first.st_dev, first.st_ino, first.st_mtime_ns, first.st_ctime_ns):
        raise ValueError("model path changed")
    if digest.hexdigest() != expected_sha256:
        raise ValueError("pinned GGUF SHA256 mismatch")
    return digest.hexdigest()


def preflight(model: Path, binary: Path, *, enforce_runtime: bool = False) -> dict:
    reasons: list[str] = []
    host = os.uname().nodename
    if host != PIN["host"]:
        reasons.append("WRONG_HOST")
    try:
        mem = memory_snapshot()
        if mem["available_bytes"] < MIN_AVAIL_BYTES:
            reasons.append("LOW_RAM")
        if mem["full_psi_10_percent"] >= MAX_FULL_PSI:
            reasons.append("MEMORY_PRESSURE")
    except (ValueError, OSError):
        mem = None
        reasons.append("UNAVAILABLE_HOST_TELEMETRY")
    try:
        gpu = gpu_snapshot()
        if gpu["free_bytes"] < PIN["bytes"] + MIN_VRAM_MARGIN:
            reasons.append("INSUFFICIENT_VRAM_RESERVE")
    except (ValueError, OSError):
        gpu = None
        reasons.append("GPU_DEVICE_UNQUALIFIED")
    if model != STAGE / "artifacts" / PIN["filename"]:
        reasons.append("MODEL_NOT_ISOLATED")
    try:
        pinned_regular(model, PIN["bytes"])
    except (ValueError, OSError):
        reasons.append("PINNED_MODEL_MISSING_OR_INCOMPLETE")
    if binary != STAGE / "bin" / "llama-bench":
        reasons.append("BENCHMARK_BINARY_NOT_ISOLATED")
    try:
        pinned_regular(binary)
        if not os.access(binary, os.X_OK):
            reasons.append("BENCH_BINARY_NOT_EXECUTABLE")
    except (ValueError, OSError):
        reasons.append("BENCH_BINARY_MISSING")
    if enforce_runtime:
        # Verifies source revision via immutable local build receipt separately.
        receipt = STAGE / "runtime-manifest.json"
        try:
            item = json.loads(receipt.read_text())
            if item["source_sha"] != PIN["runtime_source_sha"]:
                reasons.append("RUNTIME_SOURCE_NOT_PINNED")
            if hash_pinned(binary, int(item["binary_bytes"]), item["binary_sha256"]) is None:
                reasons.append("RUNTIME_SHA_UNVERIFIED")
        except (OSError, ValueError, KeyError, TypeError):
            reasons.append("RUNTIME_MANIFEST_UNVERIFIED")
    return {"schema": PIN["schema"], "captured_utc": utc(), "host": host,
            "quant": PIN["quant"], "model_file_expected_bytes": PIN["bytes"],
            "hf_revision": PIN["hf_revision"], "model_sha256_pin": PIN["sha256"],
            "runtime_source_sha": PIN["runtime_source_sha"],
            "memory": mem, "gpu": gpu, "reasons": reasons,
            "status": "READY_TO_VALIDATE_HASH" if not reasons else "HOLD",
            "production_admission": "NO_GO"}


def parse_bench_jsonl(text: str) -> dict:
    """Parse actual llama-bench JSONL without fabricating missing metrics."""
    observations = []
    for line in text.splitlines():
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if not isinstance(obj, dict):
            continue
        score = obj.get("avg_ts", obj.get("tps"))
        pp = obj.get("n_prompt", obj.get("n_pp"))
        tg = obj.get("n_gen", obj.get("n_tg"))
        if (type(score) not in (int, float) or not (0 < score < 1e6) or
            type(pp) not in (int, float) or type(tg) not in (int, float)):
            continue
        observations.append({
            "n_prompt": int(pp), "n_gen": int(tg),
            "avg_tokens_per_second": round(float(score), 3),
            "repetitions": obj.get("n_repetitions"),
            "device_info": str(obj.get("gpu_info", ""))[:128],
        })
    return {"valid_results": len(observations), "observations": observations}


def write_create_only(path: Path, content: str) -> None:
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise ValueError("unsafe result directory")
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY |
                 getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def measure(model: Path, binary: Path, profile: str, output: Path) -> dict:
    record = preflight(model, binary, enforce_runtime=True)
    record.update({"profile": profile, "benchmark_attempted": False,
                   "model_hash_verified": False, "elapsed_seconds": None,
                   "exit_code": None, "bench": None, "stopped_reason": None,
                   "monitor_samples": [],
                   "monitor_summary": {"min_available_gib": None,
                                       "max_full_psi_10_percent": None,
                                       "max_vram_used_gib": None},
                   "production_admission": "NO_GO"})
    if record["reasons"]:
        return record
    try:
        hash_pinned(model, PIN["bytes"], PIN["sha256"])
        record["model_hash_verified"] = True
    except (ValueError, OSError):
        record["reasons"].append("MODEL_SHA256_FAILED")
        record["status"] = "HOLD"
        return record
    # Hashing 14.5 GB briefly warms page cache and can depress MemAvailable
    # without any GPU activity. Retry only a bounded number of *read-only*
    # checks, without relaxing RAM/PSI/GPU limits or touching services.
    record["posthash_checks"] = []
    for attempt in range(6):
        check = preflight(model, binary, enforce_runtime=True)
        record["posthash_checks"].append({
            "attempt": attempt + 1,
            "memory": check.get("memory"),
            "gpu": check.get("gpu"),
            "reasons": check["reasons"],
        })
        if not check["reasons"]:
            break
        if any(reason not in {"LOW_RAM", "MEMORY_PRESSURE"}
               for reason in check["reasons"]) or attempt == 5:
            record["reasons"].append("RESOURCES_CHANGED_AFTER_SHA")
            record["status"] = "HOLD"
            return record
        time.sleep(2)
    # --device Vulkan1, not default CPU-integrated Vulkan0.
    cfg = PROFILES[profile]
    args = [str(binary), "-m", str(model),
            "-dev", PIN["device"], "-ngl", "99",
            "-p", cfg["n_prompt"], "-n", cfg["n_gen"],
            "-r", str(cfg["r"]), "-b", "128", "-ub", "128",
            "-t", "6", "-fa", "on", "-o", "jsonl"]
    unit = "flint-r9700-" + uuid.uuid4().hex[:12] + ".scope"
    # User-owned, transient cgroup. Never calls a production service manager.
    wrapped = ["systemd-run", "--user", "--scope", "--collect",
               "--unit=" + unit, "-p", "MemoryAccounting=yes",
               "-p", "MemoryHigh=2500M", "-p", "MemoryMax=3500M",
               "-p", "TasksMax=96", "--"] + args
    # Device enumeration on Joyner is Vulkan0=integrated and Vulkan1=R9700.
    # Do not filter/remap indices while also passing explicit -dev Vulkan1.
    env = dict(os.environ, LD_LIBRARY_PATH=str(binary.parent))
    env.pop("GGML_VK_VISIBLE_DEVICES", None)
    record["benchmark_attempted"] = True
    record["runtime_argv_labels"] = {"gpu": "Vulkan1", "threads": 6,
                                      "n_gpu_layers": 99, "flash_attention": "on"}
    t0 = time.monotonic()
    proc = None
    try:
        # Disk-backed output prevents loader diagnostics filling OS pipes and
        # deadlocking the memory/time safety monitor.
        outlog = tempfile.TemporaryFile(mode="w+b")
        errlog = tempfile.TemporaryFile(mode="w+b")
        proc = subprocess.Popen(wrapped, stdout=outlog,
                                stderr=errlog, env=env,
                                start_new_session=True)
        # Only the unique experiment unit can be stopped. Scope cleanup
        # also covers grandchildren if the launcher aborts.
        while proc.poll() is None:
            time.sleep(1)
            if time.monotonic() - t0 > cfg["max_sec"]:
                record["stopped_reason"] = "TIME_LIMIT"
                break
            try:
                m = memory_snapshot()
                g = gpu_snapshot()
                row = {
                    "elapsed_seconds": round(time.monotonic() - t0, 3),
                    "available_gib": m["available_gib"],
                    "full_psi_10_percent": m["full_psi_10_percent"],
                    "vram_used_gib": g["used_gib"],
                    "gpu_power": g["runtime_power"],
                }
                record["monitor_samples"].append(row)
                summary = record["monitor_summary"]
                summary["min_available_gib"] = min(
                    v for v in (summary["min_available_gib"], row["available_gib"])
                    if v is not None)
                summary["max_full_psi_10_percent"] = max(
                    v for v in (summary["max_full_psi_10_percent"],
                                row["full_psi_10_percent"]) if v is not None)
                summary["max_vram_used_gib"] = max(
                    v for v in (summary["max_vram_used_gib"],
                                row["vram_used_gib"]) if v is not None)
                if m["available_bytes"] < ABORT_AVAIL_BYTES or m["full_psi_10_percent"] >= 5.0:
                    record["stopped_reason"] = "HOST_MEMORY_SAFETY"
                if g["free_bytes"] < 256 * 1024 ** 2:
                    record["stopped_reason"] = "GPU_VRAM_EXHAUSTION"
            except (ValueError, OSError):
                record["stopped_reason"] = "RUNTIME_TELEMETRY_LOST"
            if record["stopped_reason"]:
                break
        if record["stopped_reason"]:
            subprocess.run(["systemctl", "--user", "stop", unit],
                           timeout=10, check=False, capture_output=True)
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        try:
            proc.wait(timeout=12)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=8)
        outlog.seek(0)
        errlog.seek(0)
        stdout = outlog.read(2 * 1024 * 1024).decode("utf-8", "replace")
        stderr = errlog.read(2 * 1024 * 1024).decode("utf-8", "replace")
        record["logs_truncated"] = (
            outlog.read(1) != b"" or errlog.read(1) != b""
        )
        outlog.close()
        errlog.close()
        record["exit_code"] = proc.returncode
        record["elapsed_seconds"] = round(time.monotonic() - t0, 3)
        record["bench"] = parse_bench_jsonl(stdout)
        record["stderr_sha256"] = hashlib.sha256(stderr.encode()).hexdigest()
        record["stderr_bytes"] = len(stderr)
        if proc.returncode == 0 and record["bench"]["valid_results"] > 0 and not record["stopped_reason"]:
            record["status"] = "MEASURED_ISOLATED"
        else:
            record["status"] = "BLOCKED_OR_FAILED"
            record["reasons"].append("BENCHMARK_EXIT_OR_TIMINGS_INVALID")
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        record["status"] = "BLOCKED_OR_FAILED"
        record["reasons"].append("BENCHMARK_LAUNCH_FAILED_" + type(exc).__name__)
    return record


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--quant", choices=sorted(QUANTS), default="IQ2_XXS")
    ap.add_argument("--profile", choices=sorted(PROFILES), default="smoke")
    args = ap.parse_args(argv)
    PIN.update(QUANTS[args.quant])
    model = STAGE / "artifacts" / PIN["filename"]
    binary = STAGE / "bin" / "llama-bench"
    if not args.run:
        record = preflight(model, binary)
        print(json.dumps(record, indent=2))
        return 2
    if os.environ.get("ALLOW_FLINT_R9700_BENCH") != "1":
        print(json.dumps({"status":"HOLD","reason":"EXPLICIT_BENCH_OPT_IN_MISSING",
                          "production_admission":"NO_GO"}))
        return 2
    output = STAGE / "receipts"
    if not output.is_dir():
        raise ValueError("private experiment receipt directory absent")
    receipt = output / ("flint-" + args.quant.lower() + "-" + args.profile + "-" +
                        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") +
                        "-" + uuid.uuid4().hex[:8] + ".json")
    report = measure(model, binary, args.profile, receipt)
    write_create_only(receipt, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "reason": report["reasons"],
                      "attempted": report["benchmark_attempted"],
                      "actual_metrics": report.get("bench"),
                      "receipt": str(receipt),
                      "production_admission": "NO_GO"}, indent=2))
    return 2  # Measurements are not fleet production admission.


if __name__ == "__main__":
    raise SystemExit(main())
