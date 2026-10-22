"""Prepare and optionally execute the exact R9700/Bonsai 2 qualification lane."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence
from urllib.parse import urlparse

import requests

from lms_agent_bench.hermes_agent_common import canonical_hash, require_loopback_endpoint
from lms_agent_bench.model_loadout import validate_manifest

NODE_ID = "x1-370"
GPU_ARCH = "gfx1201"
MODEL_ID = "Ternary-Bonsai-2-27B-PQ2_0"
MODEL_REPO = "prism-ml/Ternary-Bonsai-2-27B-gguf"
MODEL_REVISION = "6ed5e12bf84b7a63069882c91dd9e9218647d17b"
MODEL_SHA256 = "sha256:3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1"
MODEL_PARAMETERS = 26895998464
MODEL_FTYPE_PREFIX = "PQ2_0"
RUNTIME_REF = "PrismML-Eng/llama.cpp@9a9394a895b96003ca842a6041cb28ac49a108f7"
RUNTIME_TAG = "prism-b10709-9a9394a"
RUNTIME_BUILD_MARKER = "b10709-9a9394a89"
RUNTIME_SHA256 = "sha256:e5c4211999de5b789b980626ad4b35f17d9b75f9a45c6d8b1801ff41f9162c88"
DEFAULT_MODEL = (
    "~/.local/share/local-studio/experimental/bonsai2-r9700/models/"
    "Ternary-Bonsai-2-27B-PQ2_0.gguf"
)
DEFAULT_ENDPOINT = "http://127.0.0.1:8000/v1"
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    if process.returncode != 0:
        raise ValueError(
            f"git {' '.join(args)} failed for {repo}: {process.stderr.strip()}"
        )
    return process.stdout.strip()


def source_identity(
    repo: Path,
    *,
    label: str,
    expected_branch: str,
    expected_commit: str,
) -> Dict[str, str]:
    resolved = repo.expanduser().resolve()
    commit = str(expected_commit or "").lower()
    if not _COMMIT_RE.fullmatch(commit):
        raise ValueError(f"{label} expected commit must be 40 lowercase hex characters")
    if _git(resolved, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError(f"{label} checkout is not completely clean")
    branch = _git(resolved, "branch", "--show-current")
    actual = _git(resolved, "rev-parse", "HEAD").lower()
    if branch != expected_branch:
        raise ValueError(
            f"{label} branch mismatch: expected {expected_branch}, found {branch}"
        )
    if actual != commit:
        raise ValueError(
            f"{label} commit mismatch: expected {commit}, found {actual}"
        )
    return {"repo": str(resolved), "branch": branch, "commit": actual}


def _option_value(argv: Sequence[str], *names: str) -> Optional[str]:
    for index, value in enumerate(argv):
        for name in names:
            if value == name:
                if index + 1 >= len(argv):
                    raise ValueError(f"runtime option {name} has no value")
                return argv[index + 1]
            prefix = name + "="
            if value.startswith(prefix):
                return value[len(prefix) :]
    return None


def _bool_option(argv: Sequence[str], *names: str) -> Optional[bool]:
    value = _option_value(argv, *names)
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"runtime boolean option has unsupported value: {value}")


def parse_runtime_arguments(
    argv: Sequence[str],
    *,
    model_path: Path,
    endpoint: str,
) -> Dict[str, Any]:
    if not argv:
        raise ValueError("runtime process has no argv")
    model = _option_value(argv, "-m", "--model")
    if model is None or Path(model).expanduser().resolve() != model_path.resolve():
        raise ValueError("runtime argv does not reference the exact model artifact")
    endpoint_parsed = urlparse(require_loopback_endpoint(endpoint))
    expected_port = endpoint_parsed.port or (443 if endpoint_parsed.scheme == "https" else 80)
    port_value = _option_value(argv, "--port")
    if port_value is None or int(port_value) != expected_port:
        raise ValueError("runtime argv port does not match qualification endpoint")
    host = _option_value(argv, "--host")
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("runtime must bind loopback for qualification")
    gpu_layers = _option_value(argv, "-ngl", "--gpu-layers")
    if gpu_layers is None or int(gpu_layers) != 999:
        raise ValueError("Bonsai qualification requires the accepted 999 GPU-layer loadout")
    flash_attention = _bool_option(argv, "-fa", "--flash-attn")
    if flash_attention is not True:
        raise ValueError("Bonsai qualification requires flash attention on")
    context = _option_value(argv, "-c", "--ctx-size")
    parallel = _option_value(argv, "-np", "--parallel")
    batch = _option_value(argv, "-b", "--batch-size")
    ubatch = _option_value(argv, "-ub", "--ubatch-size")
    threads = _option_value(argv, "-t", "--threads")
    return {
        "gpu_layers": 999,
        "flash_attention": True,
        "context_tokens": int(context) if context else None,
        "parallel_slots": int(parallel) if parallel else None,
        "batch_size": int(batch) if batch else None,
        "ubatch_size": int(ubatch) if ubatch else None,
        "threads": int(threads) if threads else None,
        "argv_sha256": "sha256:"
        + hashlib.sha256(b"\0".join(item.encode("utf-8") for item in argv)).hexdigest(),
    }


def find_runtime_process(model_path: Path, endpoint: str) -> Dict[str, Any]:
    candidates = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            raw = (proc / "cmdline").read_bytes()
            argv = [
                item.decode("utf-8")
                for item in raw.split(b"\0")
                if item
            ]
            if not argv:
                continue
            observed = parse_runtime_arguments(
                argv,
                model_path=model_path,
                endpoint=endpoint,
            )
            executable = (proc / "exe").resolve()
        except (OSError, UnicodeDecodeError, ValueError):
            continue
        candidates.append(
            {
                "pid": int(proc.name),
                "argv": argv,
                "executable": executable,
                "arguments": observed,
            }
        )
    if len(candidates) != 1:
        raise ValueError(
            "expected exactly one matching Bonsai llama-server process, "
            f"found {len(candidates)}"
        )
    return candidates[0]


def validate_props(
    props: Mapping[str, Any],
    *,
    model_path: Path,
    runtime_arguments: Mapping[str, Any],
) -> Dict[str, Any]:
    build_info = str(props.get("build_info") or "")
    if RUNTIME_BUILD_MARKER not in build_info:
        raise ValueError("endpoint build_info does not match the accepted PrismML build")
    alias = Path(str(props.get("model_alias") or "")).expanduser().resolve()
    if alias != model_path.resolve():
        raise ValueError("endpoint model_alias does not match the exact model artifact")
    ftype = str(props.get("model_ftype") or "")
    if not ftype.startswith(MODEL_FTYPE_PREFIX):
        raise ValueError("endpoint model_ftype is not the accepted PQ2_0 artifact")
    generation = props.get("default_generation_settings")
    if not isinstance(generation, Mapping):
        raise ValueError("endpoint props omit default_generation_settings")
    context_tokens = int(generation.get("n_ctx") or 0)
    if context_tokens <= 0:
        raise ValueError("endpoint props omit a positive n_ctx")
    process_context = runtime_arguments.get("context_tokens")
    if process_context is not None and int(process_context) != context_tokens:
        raise ValueError("runtime argv context and endpoint props n_ctx disagree")
    slots = int(props.get("total_slots") or 0)
    if slots <= 0:
        raise ValueError("endpoint props omit a positive total_slots")
    process_slots = runtime_arguments.get("parallel_slots")
    if process_slots is not None and int(process_slots) != slots:
        raise ValueError("runtime argv parallel slots and endpoint props disagree")
    params = generation.get("params")
    if isinstance(params, Mapping):
        speculative = str(params.get("speculative.types") or "none").lower()
        if speculative not in {"", "none"}:
            raise ValueError("physical baseline must not use speculative decoding")
    modalities = props.get("modalities")
    vision = bool(modalities.get("vision")) if isinstance(modalities, Mapping) else False
    return {
        "build_info": build_info,
        "context_tokens": context_tokens,
        "parallel_slots": slots,
        "vision_projector_loaded": vision,
        "props_fingerprint": canonical_hash(dict(props)),
    }


def build_loadout(
    *,
    model_path: Path,
    runtime_path: Path,
    runtime_arguments: Mapping[str, Any],
    props_observation: Mapping[str, Any],
) -> Dict[str, Any]:
    runtime: Dict[str, Any] = {
        "engine": "PrismML llama.cpp",
        "engine_version": RUNTIME_REF,
        "engine_build": RUNTIME_TAG,
        "engine_content_sha256": RUNTIME_SHA256,
        "backend": "rocm",
        "gpu_layers": int(runtime_arguments["gpu_layers"]),
        "flash_attention": bool(runtime_arguments["flash_attention"]),
        "engine_arguments": [],
        "observed_process_argv_sha256": runtime_arguments["argv_sha256"],
        "runtime_binary": str(runtime_path),
        "rocm_version": "7.2.0",
        "gpu_arch": GPU_ARCH,
        "vision_projector_loaded": bool(
            props_observation["vision_projector_loaded"]
        ),
    }
    for source, target in (
        ("batch_size", "batch_size"),
        ("ubatch_size", "ubatch_size"),
        ("threads", "threads"),
    ):
        value = runtime_arguments.get(source)
        if value is not None:
            runtime[target] = int(value)
    raw = {
        "schema_version": "model_loadout_manifest.v1",
        "node_id": NODE_ID,
        "candidate_id": "r9700-bonsai2-prism-rocm-text-baseline",
        "model": {
            "id": MODEL_ID,
            "content_sha256": MODEL_SHA256,
            "format": "gguf",
            "size_bytes": model_path.stat().st_size,
            "family": "ternary-bonsai-2",
            "revision": MODEL_REVISION,
            "source_repo": MODEL_REPO,
        },
        "architecture": {
            "kind": "other",
            "architecture_name": "ternary-hybrid",
            "total_parameter_count": MODEL_PARAMETERS,
            "attention_type": "hybrid",
        },
        "weight_quantization": {
            "scheme": "PQ2_0",
            "nominal_bits": 2.13,
            "effective_bits_per_weight": 2.13,
            "group_size": 128,
            "mixed_precision": False,
        },
        "runtime": runtime,
        "context": {
            "configured_tokens": int(props_observation["context_tokens"]),
            "prompt_tokens_target": min(
                4096,
                int(props_observation["context_tokens"]),
            ),
        },
        "kv_cache": {
            "key_dtype": "runtime-default",
            "value_dtype": "runtime-default",
            "location": "unknown",
            "capacity_tokens": int(props_observation["context_tokens"]),
            "shared_across_requests": False,
            "prefix_reuse": False,
            "persistent": False,
            "cache_policy": "runtime-default",
        },
        "concurrency": {
            "parallel_slots": int(props_observation["parallel_slots"]),
            "continuous_batching": True,
            "max_queued_requests": None,
        },
        "speculative_decoding": {"enabled": False},
        "physical_evidence": {
            "accepted_lane": "r9700-bonsai2-prism-rocm",
            "accepted_model_sha256": MODEL_SHA256,
            "accepted_runtime_sha256": RUNTIME_SHA256,
            "endpoint_props_fingerprint": props_observation["props_fingerprint"],
        },
    }
    return validate_manifest(raw)


def _json_get(url: str, timeout: float) -> Mapping[str, Any]:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object from {url}")
    return payload


def endpoint_observation(
    endpoint: str,
    *,
    model_path: Path,
    runtime_arguments: Mapping[str, Any],
    timeout: float,
) -> Dict[str, Any]:
    base = require_loopback_endpoint(endpoint).rstrip("/")
    models = _json_get(base + "/models", timeout)
    ids = [
        str(item.get("id"))
        for item in models.get("data", [])
        if isinstance(item, Mapping) and item.get("id")
    ]
    if ids.count(MODEL_ID) != 1:
        raise ValueError("endpoint does not expose the exact Bonsai model exactly once")
    parsed = urlparse(base)
    root = f"{parsed.scheme}://{parsed.netloc}"
    props = _json_get(root + "/props", timeout)
    validated = validate_props(
        props,
        model_path=model_path,
        runtime_arguments=runtime_arguments,
    )
    return {
        **validated,
        "endpoint": base,
        "models_fingerprint": canonical_hash(models),
    }


def hardware_observation() -> Dict[str, Any]:
    host = platform.node().split(".", 1)[0]
    if host != NODE_ID:
        raise ValueError(f"physical proof must run on {NODE_ID}, found {host}")
    process = subprocess.run(
        ["rocminfo"],
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    if process.returncode != 0:
        raise ValueError("rocminfo failed during physical preflight")
    if GPU_ARCH not in process.stdout:
        raise ValueError(f"rocminfo did not report required architecture {GPU_ARCH}")
    return {
        "node_id": host,
        "gpu_arch": GPU_ARCH,
        "rocminfo_sha256": "sha256:"
        + hashlib.sha256(process.stdout.encode("utf-8")).hexdigest(),
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def prepare(args: argparse.Namespace) -> Dict[str, Any]:
    endpoint = require_loopback_endpoint(args.endpoint)
    model_path = args.model_artifact.expanduser().resolve()
    if not model_path.is_file() or model_path.is_symlink():
        raise ValueError("model artifact must be a regular non-symlink file")
    if file_sha256(model_path) != MODEL_SHA256:
        raise ValueError("Bonsai model SHA-256 does not match accepted evidence")

    lms_repo = args.lms_repo.expanduser().resolve()
    hermes_repo = args.hermes_repo.expanduser().resolve()
    lms_source = source_identity(
        lms_repo,
        label="LMS",
        expected_branch=args.lms_branch,
        expected_commit=args.lms_commit,
    )
    hermes_source = source_identity(
        hermes_repo,
        label="Hermes",
        expected_branch=args.hermes_branch,
        expected_commit=args.hermes_commit,
    )
    if not (hermes_repo / "run_agent.py").is_file():
        raise ValueError("Hermes checkout lacks run_agent.py")

    hardware = hardware_observation()
    runtime_process = find_runtime_process(model_path, endpoint)
    runtime_path = Path(runtime_process["executable"]).resolve()
    if file_sha256(runtime_path) != RUNTIME_SHA256:
        raise ValueError("live runtime binary SHA-256 does not match accepted evidence")
    endpoint_state = endpoint_observation(
        endpoint,
        model_path=model_path,
        runtime_arguments=runtime_process["arguments"],
        timeout=args.timeout,
    )
    loadout = build_loadout(
        model_path=model_path,
        runtime_path=runtime_path,
        runtime_arguments=runtime_process["arguments"],
        props_observation=endpoint_state,
    )

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    loadout_path = output / "loadout.json"
    inventory_path = output / "inventory.csv"
    cases_path = output / "throughput-cases.json"
    preflight_path = output / "physical-preflight.json"
    write_json(loadout_path, loadout)
    with inventory_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "host_name",
                "host_ip",
                "endpoint_id",
                "base_url",
                "reachable",
                "model_id",
                "model_key",
            ]
        )
        writer.writerow([NODE_ID, "127.0.0.1", "1", endpoint, "1", "1", MODEL_ID])
    source_suite = Path(__file__).resolve().parent / "benchmarks" / "agent_skill_suite.v1.json"
    shutil.copyfile(source_suite, cases_path)
    preflight = {
        "schema_version": "r9700_bonsai2_qualification_preflight.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": hardware,
        "model": {
            "path": str(model_path),
            "sha256": MODEL_SHA256,
            "size_bytes": model_path.stat().st_size,
            "revision": MODEL_REVISION,
        },
        "runtime": {
            "pid": runtime_process["pid"],
            "path": str(runtime_path),
            "sha256": RUNTIME_SHA256,
            "arguments": runtime_process["arguments"],
        },
        "endpoint": endpoint_state,
        "sources": {"lms": lms_source, "hermes": hermes_source},
        "loadout_fingerprint": loadout["loadout_fingerprint"],
        "throughput_suite_sha256": file_sha256(cases_path),
        "admission": {"admitted": False},
    }
    write_json(preflight_path, preflight)
    return {
        "output_dir": str(output),
        "loadout": str(loadout_path),
        "inventory": str(inventory_path),
        "cases": str(cases_path),
        "preflight": str(preflight_path),
        "model_artifact": str(model_path),
        "endpoint": endpoint,
        "lms_source": lms_source,
        "hermes_source": hermes_source,
        "loadout_fingerprint": loadout["loadout_fingerprint"],
    }


def operator_command(args: argparse.Namespace, prepared: Mapping[str, Any]) -> list[str]:
    return [
        sys.executable,
        "-m",
        "lms_agent_bench.loadout_qualification_operator",
        "run",
        "--loadout",
        str(prepared["loadout"]),
        "--inventory-csv",
        str(prepared["inventory"]),
        "--cases-file",
        str(prepared["cases"]),
        "--model-artifact",
        str(prepared["model_artifact"]),
        "--endpoint",
        str(prepared["endpoint"]),
        "--api-key-env",
        args.api_key_env,
        "--lms-repo",
        str(args.lms_repo.expanduser().resolve()),
        "--lms-branch",
        args.lms_branch,
        "--lms-commit",
        args.lms_commit,
        "--hermes-repo",
        str(args.hermes_repo.expanduser().resolve()),
        "--hermes-branch",
        args.hermes_branch,
        "--hermes-commit",
        args.hermes_commit,
        "--workspace",
        str(args.workspace.expanduser().resolve()),
        "--run-id",
        args.run_id,
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m lms_agent_bench.r9700_bonsai2_qualification",
        description=(
            "Fail-closed R9700/Bonsai 2 physical qualification prep and execution"
        ),
    )
    parser.add_argument("--model-artifact", type=Path, default=Path(DEFAULT_MODEL))
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--lms-repo", type=Path, required=True)
    parser.add_argument("--lms-branch", default="agent/qualification-decision-metrics")
    parser.add_argument("--lms-commit", required=True)
    parser.add_argument("--hermes-repo", type=Path, default=Path("~/git/hermes-agent"))
    parser.add_argument("--hermes-branch", default="main")
    parser.add_argument("--hermes-commit", required=True)
    parser.add_argument("--workspace", type=Path, default=Path("~/lms-qualification-runs"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--api-key-env", default="LMSTUDIO_API_KEY")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.timeout <= 0:
            raise ValueError("timeout must be positive")
        prepared = prepare(args)
        command = operator_command(args, prepared)
        prepared = {**prepared, "operator_command": command}
        print(json.dumps(prepared, indent=2, sort_keys=True))
        if not args.execute:
            return 0
        env = dict(os.environ)
        lms_src = str(args.lms_repo.expanduser().resolve() / "src")
        env["PYTHONPATH"] = lms_src + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        run = subprocess.run(command, env=env, check=False)
        if run.returncode != 0:
            return run.returncode
        verify = [
            sys.executable,
            "-m",
            "lms_agent_bench.loadout_qualification_operator",
            "verify",
            "--run-dir",
            str(args.workspace.expanduser().resolve() / args.run_id),
            "--require-success",
        ]
        return subprocess.run(verify, env=env, check=False).returncode
    except (
        OSError,
        ValueError,
        requests.RequestException,
        subprocess.SubprocessError,
    ) as exc:
        print(f"R9700/Bonsai qualification rejected: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
