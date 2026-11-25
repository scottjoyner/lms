#!/usr/bin/env python3
"""Execute an LMS fleet benchmark plan with ephemeral, loopback-only runtimes."""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    import requests
except ImportError as exc:  # pragma: no cover
    raise SystemExit("fleet plan execution requires the package dependency 'requests'") from exc

from lms_agent_bench.fleet_loadout import canonical_hash, load_json, write_json

DEFAULT_STARTUP_TIMEOUT_S = 120
DEFAULT_REQUEST_TIMEOUT_S = 180


def utc_now_iso() -> str:
    import datetime as dt

    return dt.datetime.now(dt.timezone.utc).isoformat()


def normalize_base_url(value: str) -> str:
    value = value.strip().rstrip("/")
    return value if value.endswith("/v1") else value + "/v1"


def safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in value).strip("-") or "candidate"


def candidate_map(plan: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(item["candidate_id"]): dict(item)
        for item in plan.get("candidates", [])
        if isinstance(item, dict) and item.get("candidate_id")
    }


def resolve_candidates(
    plan: Mapping[str, Any], requested: Sequence[str], run_all: bool, limit: int
) -> List[Dict[str, Any]]:
    by_id = candidate_map(plan)
    if requested:
        unknown = sorted(set(requested) - set(by_id))
        if unknown:
            raise ValueError(f"unknown candidate IDs: {', '.join(unknown)}")
        selected = [by_id[candidate_id] for candidate_id in requested]
    elif run_all:
        selected = list(by_id.values())
    else:
        raise ValueError("choose at least one --candidate or pass --all")
    return selected[:limit] if limit > 0 else selected


def parse_endpoint_map(values: Sequence[str]) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--endpoint-map values must be CANDIDATE_ID=URL")
        candidate_id, url = value.split("=", 1)
        candidate_id, url = candidate_id.strip(), url.strip()
        if not candidate_id or not url:
            raise ValueError("--endpoint-map values must be CANDIDATE_ID=URL")
        parsed[candidate_id] = normalize_base_url(url)
    return parsed


def command_supported(binary: str, timeout_s: int = 10) -> str:
    try:
        proc = subprocess.run(
            [binary, "--help"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        return (proc.stdout or "") + "\n" + (proc.stderr or "")
    except Exception:
        return ""


def add_flag(
    command: List[str],
    help_text: str,
    preferred: str,
    value: Optional[Any] = None,
    aliases: Sequence[str] = (),
) -> None:
    options = (preferred, *aliases)
    chosen = next((option for option in options if not help_text or option in help_text), None)
    if chosen is None:
        return
    command.append(chosen)
    if value is not None:
        command.append(str(value))


def build_llama_server_command(
    candidate: Mapping[str, Any], binary: str, help_text: str = ""
) -> List[str]:
    model = candidate.get("model", {})
    model_path = model.get("path")
    if not model_path:
        raise ValueError(f"candidate {candidate.get('candidate_id')} model has no path")
    if not Path(str(model_path)).exists():
        raise ValueError(f"model path does not exist: {model_path}")
    port = int(candidate.get("benchmark_port") or 18080)
    command = [binary]
    add_flag(command, help_text, "--model", model_path, aliases=("-m",))
    add_flag(command, help_text, "--host", "127.0.0.1")
    add_flag(command, help_text, "--port", port)
    add_flag(
        command,
        help_text,
        "--ctx-size",
        int(candidate.get("context_tokens") or 4096),
        aliases=("-c",),
    )
    add_flag(
        command,
        help_text,
        "--parallel",
        int(candidate.get("parallel_slots") or 1),
        aliases=("-np",),
    )
    add_flag(
        command,
        help_text,
        "--threads",
        int(candidate.get("threads") or max(1, os.cpu_count() or 1)),
        aliases=("-t",),
    )
    add_flag(
        command,
        help_text,
        "--n-gpu-layers",
        int(candidate.get("gpu_layers") or 0),
        aliases=("-ngl",),
    )
    add_flag(
        command,
        help_text,
        "--batch-size",
        int(candidate.get("batch_size") or 256),
        aliases=("-b",),
    )
    add_flag(
        command,
        help_text,
        "--ubatch-size",
        int(candidate.get("ubatch_size") or 128),
        aliases=("-ub",),
    )
    if candidate.get("flash_attention") is not None:
        add_flag(
            command,
            help_text,
            "--flash-attn",
            "on" if candidate.get("flash_attention") else "off",
            aliases=("-fa",),
        )
    for argument in candidate.get("engine_arguments", []) or []:
        command.append(str(argument))
    return command


def launch_environment(candidate: Mapping[str, Any]) -> Dict[str, str]:
    environment = dict(os.environ)
    for source in (
        candidate.get("environment", {}),
        candidate.get("launch", {}).get("environment", {}),
    ):
        if isinstance(source, Mapping):
            environment.update({str(key): str(value) for key, value in source.items()})
    if candidate.get("backend") == "vulkan" and candidate.get("vulkan_icd_filename"):
        environment["VK_ICD_FILENAMES"] = str(candidate["vulkan_icd_filename"])
    return environment


def http_json(url: str, timeout_s: float) -> Tuple[Any, Optional[str]]:
    try:
        request = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return (
                json.loads(response.read(2_000_000).decode("utf-8", errors="replace")),
                None,
            )
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return None, repr(exc)


def wait_for_endpoint(
    base_url: str, process: Optional[subprocess.Popen[Any]], timeout_s: float
) -> Dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last_error = "not attempted"
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            return {"ready": False, "error": f"process exited with code {process.returncode}"}
        data, error = http_json(
            base_url.rstrip("/") + "/models", timeout_s=min(3.0, timeout_s)
        )
        if error is None and isinstance(data, dict):
            models = [
                str(item.get("id"))
                for item in data.get("data", [])
                if isinstance(item, dict) and item.get("id")
            ]
            return {"ready": True, "models": models, "error": None}
        last_error = error or "invalid response"
        time.sleep(0.5)
    return {"ready": False, "error": f"startup timeout: {last_error}"}


def process_rss_bytes(pid: int) -> Optional[int]:
    if platform.system() == "Linux":
        try:
            for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            return None
    try:
        proc = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        return (
            int(proc.stdout.strip()) * 1024
            if proc.returncode == 0 and proc.stdout.strip()
            else None
        )
    except Exception:
        return None


def system_memory() -> Tuple[Optional[int], Optional[int]]:
    if platform.system() == "Linux":
        values: Dict[str, int] = {}
        try:
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                if ":" in line:
                    key, rest = line.split(":", 1)
                    token = rest.strip().split()[0]
                    values[key] = int(token) * 1024
            return values.get("MemTotal"), values.get("MemAvailable")
        except (OSError, ValueError, IndexError):
            return None, None
    if platform.system() == "Darwin":
        try:
            total = int(
                subprocess.check_output(
                    ["sysctl", "-n", "hw.memsize"], text=True, timeout=3
                ).strip()
            )
            page_size = int(
                subprocess.check_output(
                    ["sysctl", "-n", "hw.pagesize"], text=True, timeout=3
                ).strip()
            )
            vm = subprocess.check_output(["vm_stat"], text=True, timeout=3)
            pages = 0
            for line in vm.splitlines():
                if any(
                    line.startswith(prefix)
                    for prefix in ("Pages free", "Pages inactive", "Pages speculative")
                ):
                    pages += int(line.split(":", 1)[1].strip().rstrip("."))
            return total, pages * page_size
        except Exception:
            return None, None
    return None, None


class ResourceSampler:
    def __init__(self, pid: Optional[int], interval_s: float = 0.25):
        self.pid = pid
        self.interval_s = interval_s
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.peak_rss_bytes = 0
        self.minimum_available_bytes: Optional[int] = None
        self.total_memory_bytes: Optional[int] = None

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while not self.stop_event.is_set():
            if self.pid:
                rss = process_rss_bytes(self.pid)
                if rss:
                    self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
            total, available = system_memory()
            if total:
                self.total_memory_bytes = total
            if available is not None:
                self.minimum_available_bytes = (
                    available
                    if self.minimum_available_bytes is None
                    else min(self.minimum_available_bytes, available)
                )
            self.stop_event.wait(self.interval_s)

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2)

    def headroom_ratio(self) -> Optional[float]:
        if self.total_memory_bytes and self.minimum_available_bytes is not None:
            return self.minimum_available_bytes / self.total_memory_bytes
        return None


def post_chat(
    base_url: str,
    model: str,
    stream: bool,
    timeout_s: float,
    max_tokens: int = 32,
) -> requests.Response:
    return requests.post(
        base_url.rstrip("/") + "/chat/completions",
        json={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "Reply with the word READY and nothing else.",
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": stream,
        },
        stream=stream,
        timeout=timeout_s,
    )


def streaming_gate(base_url: str, model: str, timeout_s: float) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        with post_chat(base_url, model, stream=True, timeout_s=timeout_s) as response:
            if response.status_code >= 400:
                return {
                    "ok": False,
                    "status": response.status_code,
                    "error": response.text[:500],
                }
            content_seen = False
            done_seen = False
            for raw in response.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data:"):
                    continue
                data = raw[5:].strip()
                if data == "[DONE]":
                    done_seen = True
                    break
                try:
                    chunk = json.loads(data)
                    delta = ((chunk.get("choices") or [{}])[0].get("delta") or {})
                    content_seen = content_seen or bool(delta.get("content"))
                except Exception:
                    continue
            return {
                "ok": content_seen and done_seen,
                "content_seen": content_seen,
                "done_seen": done_seen,
                "wall_s": time.perf_counter() - started,
            }
    except Exception as exc:
        return {
            "ok": False,
            "error": repr(exc),
            "wall_s": time.perf_counter() - started,
        }


def concurrency_gate(
    base_url: str, model: str, parallel_slots: int, timeout_s: float
) -> Dict[str, Any]:
    workers = max(2, int(parallel_slots) + 1)
    started = time.perf_counter()

    def call_one(index: int) -> Dict[str, Any]:
        t0 = time.perf_counter()
        try:
            response = post_chat(
                base_url, model, stream=False, timeout_s=timeout_s, max_tokens=16
            )
            return {
                "index": index,
                "ok": response.status_code < 400,
                "status": response.status_code,
                "wall_s": time.perf_counter() - t0,
            }
        except Exception as exc:
            return {
                "index": index,
                "ok": False,
                "error": repr(exc),
                "wall_s": time.perf_counter() - t0,
            }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(call_one, index) for index in range(workers)]
        results = [future.result() for future in as_completed(futures)]
    return {
        "ok": all(item["ok"] for item in results),
        "workers": workers,
        "wall_s": time.perf_counter() - started,
        "results": sorted(results, key=lambda item: item["index"]),
    }


def cancellation_gate(base_url: str, model: str, timeout_s: float) -> Dict[str, Any]:
    try:
        response = post_chat(
            base_url, model, stream=True, timeout_s=timeout_s, max_tokens=256
        )
        first_content = False
        for raw in response.iter_lines(decode_unicode=True):
            if raw and raw.startswith("data:") and raw[5:].strip() not in {"", "[DONE]"}:
                first_content = True
                break
        response.close()
        canary_started = time.perf_counter()
        canary = post_chat(
            base_url,
            model,
            stream=False,
            timeout_s=min(timeout_s, 30),
            max_tokens=8,
        )
        return {
            "ok": first_content and canary.status_code < 400,
            "first_content": first_content,
            "canary_status": canary.status_code,
            "canary_wall_s": time.perf_counter() - canary_started,
        }
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}


def write_inventory(path: Path, base_url: str, model: str, candidate_id: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "host_name",
                "host_ip",
                "endpoint_id",
                "base_url",
                "reachable",
                "model_id",
                "model_key",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "host_name": socket.gethostname(),
                "host_ip": "127.0.0.1",
                "endpoint_id": candidate_id,
                "base_url": base_url,
                "reachable": 1,
                "model_id": 1,
                "model_key": model,
            }
        )


def run_lms_suite(
    candidate_dir: Path,
    base_url: str,
    model: str,
    candidate_id: str,
    suite_file: str,
    timeout_s: float,
    repeats: int,
    max_context_tokens: int,
) -> int:
    inventory = candidate_dir / "inventory.csv"
    write_inventory(inventory, base_url, model, candidate_id)
    output_dir = candidate_dir / "suite"
    sidecar_dir = candidate_dir / "sidecars"
    command = [
        sys.executable,
        "-m",
        "lms_agent_bench.benchmark_lmstudio_cross_machine_models",
        "--inventory-csv",
        str(inventory),
        "--cases-file",
        suite_file,
        "--output-dir",
        str(output_dir),
        "--sidecar-dir",
        str(sidecar_dir),
        "--timeout",
        str(timeout_s),
        "--repeats",
        str(repeats),
        "--max-context-tokens",
        str(max_context_tokens),
    ]
    (candidate_dir / "suite_command.json").write_text(
        json.dumps(command, indent=2) + "\n", encoding="utf-8"
    )
    with (candidate_dir / "suite.log").open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    return int(proc.returncode)


def first_csv_row(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return next(iter(csv.DictReader(handle)), {})


def terminate_process(
    process: Optional[subprocess.Popen[Any]], grace_s: float = 10
) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        process.terminate()
    try:
        process.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            process.kill()
        process.wait(timeout=5)


def execute_candidate(
    candidate: Mapping[str, Any],
    args: argparse.Namespace,
    endpoint_map: Mapping[str, str],
    help_cache: Dict[str, str],
) -> Dict[str, Any]:
    candidate_id = str(candidate["candidate_id"])
    candidate_dir = Path(args.output_dir) / safe_slug(candidate_id)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    process: Optional[subprocess.Popen[Any]] = None
    log_handle = None
    base_url = endpoint_map.get(candidate_id)
    crash_count = 0

    try:
        if base_url is None:
            if candidate.get("engine") != "llama.cpp":
                raise ValueError(
                    f"candidate {candidate_id} requires --endpoint-map because "
                    f"engine={candidate.get('engine')}"
                )
            binary = (
                args.llama_server_bin
                or os.environ.get("LLAMA_SERVER_BIN")
                or shutil.which("llama-server")
            )
            if not binary:
                raise ValueError(
                    "llama-server binary not found; pass --llama-server-bin or "
                    "LLAMA_SERVER_BIN"
                )
            help_text = help_cache.setdefault(binary, command_supported(binary))
            launch_command = build_llama_server_command(
                candidate, binary, help_text=help_text
            )
            write_json(
                str(candidate_dir / "launch.json"),
                {
                    "candidate_id": candidate_id,
                    "command": launch_command,
                    "environment_overrides": candidate.get("environment", {}),
                    "created_at_utc": utc_now_iso(),
                },
            )
            if args.dry_run:
                return {
                    "candidate_id": candidate_id,
                    "dry_run": True,
                    "launch_command": launch_command,
                }
            log_handle = (candidate_dir / "server.log").open("w", encoding="utf-8")
            process = subprocess.Popen(
                launch_command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=launch_environment(candidate),
                start_new_session=True,
                text=True,
            )
            base_url = normalize_base_url(
                f"http://127.0.0.1:{int(candidate.get('benchmark_port') or 18080)}"
            )
        else:
            base_url = normalize_base_url(base_url)

        readiness = wait_for_endpoint(base_url, process, args.startup_timeout)
        write_json(str(candidate_dir / "readiness.json"), readiness)
        if not readiness.get("ready"):
            raise RuntimeError(str(readiness.get("error")))
        model_ids = readiness.get("models") or []
        requested_model = str(candidate.get("model", {}).get("id") or "")
        model = (
            requested_model
            if requested_model in model_ids
            else (model_ids[0] if model_ids else requested_model)
        )
        if not model:
            raise RuntimeError("endpoint exposed no model ID")

        sampler = ResourceSampler(process.pid if process else None)
        sampler.start()
        try:
            streaming = streaming_gate(base_url, model, args.request_timeout)
            concurrency = concurrency_gate(
                base_url,
                model,
                int(candidate.get("parallel_slots") or 1),
                args.request_timeout,
            )
            cancellation = (
                cancellation_gate(base_url, model, args.request_timeout)
                if args.test_cancellation
                else {"ok": False, "skipped": True}
            )
            suite_exit = run_lms_suite(
                candidate_dir,
                base_url,
                model,
                candidate_id,
                args.suite_file,
                args.request_timeout,
                args.repeats,
                int(candidate.get("context_tokens") or args.max_context_tokens),
            )
        finally:
            sampler.stop()

        if process is not None and process.poll() is not None:
            crash_count = 1
        summary = first_csv_row(candidate_dir / "suite" / "run_summary.csv")
        gate_results = {
            "streaming": streaming,
            "concurrency": concurrency,
            "cancellation": cancellation,
        }
        write_json(str(candidate_dir / "gates.json"), gate_results)
        headroom = sampler.headroom_ratio()
        result = {
            "candidate_id": candidate_id,
            "engine": candidate.get("engine"),
            "backend": candidate.get("backend"),
            "model_id": model,
            "base_url": base_url,
            "ok_rate": summary.get("ok_rate", "0"),
            "eval_ok_rate": summary.get("eval_ok_rate", "0"),
            "eval_score_avg": summary.get("eval_score_avg", "0"),
            "tps_med": summary.get("tps_med", "0"),
            "ttft_med": summary.get("ttft_med", "0"),
            "memory_peak_bytes": sampler.peak_rss_bytes or "",
            "memory_headroom_ratio": (
                f"{headroom:.6f}" if headroom is not None else ""
            ),
            "concurrency_ok": concurrency.get("ok", False),
            "streaming_ok": streaming.get("ok", False),
            "cancellation_ok": cancellation.get("ok", False),
            "crash_count": crash_count,
            "benchmark_exit_code": suite_exit,
            "error": "" if suite_exit == 0 else f"LMS suite exited {suite_exit}",
            "candidate_dir": str(candidate_dir),
        }
        write_json(str(candidate_dir / "result.json"), result)
        return result
    except Exception as exc:
        if process is not None and process.poll() is not None:
            crash_count = 1
        result = {
            "candidate_id": candidate_id,
            "engine": candidate.get("engine"),
            "backend": candidate.get("backend"),
            "model_id": candidate.get("model", {}).get("id", ""),
            "base_url": base_url or "",
            "ok_rate": "0",
            "eval_ok_rate": "0",
            "eval_score_avg": "0",
            "tps_med": "0",
            "ttft_med": "0",
            "memory_peak_bytes": "",
            "memory_headroom_ratio": "",
            "concurrency_ok": False,
            "streaming_ok": False,
            "cancellation_ok": False,
            "crash_count": crash_count,
            "benchmark_exit_code": 1,
            "error": repr(exc),
            "candidate_dir": str(candidate_dir),
        }
        write_json(str(candidate_dir / "result.json"), result)
        return result
    finally:
        terminate_process(process)
        if log_handle:
            log_handle.close()
        if args.cooldown > 0 and not args.dry_run:
            time.sleep(args.cooldown)


def write_results_csv(
    path: Path, results: Sequence[Mapping[str, Any]]
) -> None:
    fields = [
        "candidate_id",
        "engine",
        "backend",
        "model_id",
        "base_url",
        "ok_rate",
        "eval_ok_rate",
        "eval_score_avg",
        "tps_med",
        "ttft_med",
        "memory_peak_bytes",
        "memory_headroom_ratio",
        "concurrency_ok",
        "streaming_ok",
        "cancellation_ok",
        "crash_count",
        "benchmark_exit_code",
        "error",
        "candidate_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result.get(field, "") for field in fields})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch and benchmark candidates from an lms-fleet plan"
    )
    parser.add_argument("--plan", required=True)
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--suite-file", required=True)
    parser.add_argument("--llama-server-bin", default=None)
    parser.add_argument("--endpoint-map", action="append", default=[])
    parser.add_argument(
        "--startup-timeout", type=float, default=DEFAULT_STARTUP_TIMEOUT_S
    )
    parser.add_argument(
        "--request-timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT_S
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-context-tokens", type=int, default=8192)
    parser.add_argument("--cooldown", type=float, default=2.0)
    parser.add_argument("--test-cancellation", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    plan = load_json(args.plan)
    candidates = resolve_candidates(plan, args.candidate, args.all, args.limit)
    endpoint_map = parse_endpoint_map(args.endpoint_map)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_ids = [item["candidate_id"] for item in candidates]
    run_manifest = {
        "schema_version": "fleet_benchmark_execution.v1",
        "created_at_utc": utc_now_iso(),
        "plan_fingerprint": plan.get("plan_fingerprint"),
        "candidate_ids": candidate_ids,
        "suite_file": args.suite_file,
        "loopback_only": True,
        "execution_fingerprint": canonical_hash(
            {
                "plan_fingerprint": plan.get("plan_fingerprint"),
                "candidate_ids": candidate_ids,
                "suite_file": args.suite_file,
            }
        ),
    }
    write_json(str(output_dir / "execution_manifest.json"), run_manifest)
    help_cache: Dict[str, str] = {}
    results = [
        execute_candidate(candidate, args, endpoint_map, help_cache)
        for candidate in candidates
    ]
    write_results_csv(output_dir / "loadout_results.csv", results)
    failures = [
        result
        for result in results
        if result.get("benchmark_exit_code") not in {0, "0"}
    ]
    print(f"wrote {output_dir / 'loadout_results.csv'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
