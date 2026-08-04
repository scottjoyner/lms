"""Execute the complete local exact-loadout qualification sequence reliably."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import requests

from lms_agent_bench import __version__
from lms_agent_bench import fleet_operator as _operator
from lms_agent_bench.fleet_operator_entrypoint import validate_run_id
from lms_agent_bench.hermes_agent_common import (
    load_json,
    require_loopback_endpoint,
)
from lms_agent_bench.loadout_qualification import verify_qualification
from lms_agent_bench.model_loadout import validate_manifest

SCHEMA_VERSION = "loadout_qualification_operator.v1"
MANIFEST_SCHEMA_VERSION = "loadout_qualification_run_manifest.v1"
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def _regular_file(path: Path, label: str) -> Path:
    value = Path(path).expanduser()
    if value.is_symlink():
        raise ValueError(f"{label} may not be a symbolic link")
    resolved = value.resolve()
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file: {resolved}")
    if os.name == "posix" and resolved.stat().st_uid != os.getuid():
        raise ValueError(f"{label} is not owned by the current user")
    return resolved


def _workspace(path: Path) -> Path:
    value = Path(path).expanduser()
    if value.is_symlink():
        raise ValueError("qualification workspace may not be a symbolic link")
    resolved = value.resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    if not os.access(resolved, os.R_OK | os.W_OK | os.X_OK):
        raise ValueError("qualification workspace is not accessible")
    return resolved


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


def source_snapshot(
    repo: Path,
    *,
    label: str,
    expected_branch: str,
    expected_commit: str,
    required_file: Optional[str] = None,
    require_contains: Optional[Path] = None,
) -> Dict[str, Any]:
    value = Path(repo).expanduser()
    if value.is_symlink():
        raise ValueError(f"{label} repository may not be a symbolic link")
    resolved = value.resolve()
    if not (resolved / ".git").is_dir():
        raise ValueError(f"{label} is not a Git checkout: {resolved}")
    commit = str(expected_commit or "").lower()
    if not _COMMIT_RE.fullmatch(commit):
        raise ValueError(f"{label} expected commit must be 40 lowercase hex characters")
    if required_file and not (resolved / required_file).is_file():
        raise ValueError(f"{label} repository lacks {required_file}")
    if require_contains is not None:
        try:
            Path(require_contains).resolve().relative_to(resolved)
        except ValueError as exc:
            raise ValueError(
                f"running package is not sourced from the reviewed {label} checkout"
            ) from exc
    status = _git(resolved, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise ValueError(f"{label} checkout is not completely clean")
    branch = _git(resolved, "branch", "--show-current")
    actual_commit = _git(resolved, "rev-parse", "HEAD").lower()
    if branch != expected_branch:
        raise ValueError(f"{label} branch mismatch: expected {expected_branch}, found {branch}")
    if actual_commit != commit:
        raise ValueError(f"{label} commit mismatch: expected {commit}, found {actual_commit}")
    origin = _git(resolved, "remote", "get-url", "origin")
    core = {
        "label": label,
        "repo": str(resolved),
        "branch": branch,
        "commit": actual_commit,
        "origin_fingerprint": "sha256:"
        + hashlib.sha256(origin.encode("utf-8")).hexdigest(),
    }
    return {**core, "source_fingerprint": _operator.canonical_hash(core)}


def inventory_identity(
    inventory_path: Path,
    loadout: Mapping[str, Any],
    endpoint: str,
) -> Dict[str, Any]:
    with inventory_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError("qualification inventory must contain exactly one endpoint/model row")
    row = rows[0]
    if str(row.get("host_name") or "") != str(loadout["node_id"]):
        raise ValueError("inventory host_name does not match loadout node_id")
    if str(row.get("model_key") or "") != str(loadout["model"]["id"]):
        raise ValueError("inventory model_key does not match loadout model ID")
    row_endpoint = require_loopback_endpoint(str(row.get("base_url") or ""))
    if row_endpoint.rstrip("/") != endpoint.rstrip("/"):
        raise ValueError("inventory base_url does not match qualification endpoint")
    return {
        "host_name": row["host_name"],
        "endpoint_id": row.get("endpoint_id"),
        "base_url": row_endpoint,
        "model_key": row["model_key"],
    }


def endpoint_probe(
    endpoint: str,
    model_id: str,
    *,
    api_key_env: str,
    timeout_seconds: float,
) -> Dict[str, Any]:
    base = require_loopback_endpoint(endpoint).rstrip("/")
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    key = os.getenv(api_key_env)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    started = time.monotonic()
    models_response = requests.get(
        base + "/models",
        headers=headers,
        timeout=timeout_seconds,
    )
    models_response.raise_for_status()
    models_payload = models_response.json()
    models = [
        str(item.get("id"))
        for item in models_payload.get("data", [])
        if isinstance(item, Mapping) and item.get("id")
    ]
    if models.count(model_id) != 1:
        raise ValueError("loopback endpoint does not expose the exact model exactly once")
    completion_started = time.monotonic()
    completion = requests.post(
        base + "/chat/completions",
        headers=headers,
        json={
            "model": model_id,
            "messages": [
                {"role": "system", "content": "You are a local health probe."},
                {"role": "user", "content": "Reply with OK."},
            ],
            "temperature": 0,
            "max_tokens": 8,
            "stream": False,
        },
        timeout=timeout_seconds,
    )
    completion.raise_for_status()
    payload = completion.json()
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("loopback completion probe returned no choices")
    message = choices[0].get("message") if isinstance(choices[0], Mapping) else None
    content = message.get("content") if isinstance(message, Mapping) else None
    if not str(content or "").strip():
        raise ValueError("loopback completion probe returned empty content")
    return {
        "endpoint": base,
        "model_id": model_id,
        "model_count": len(models),
        "models_request_seconds": completion_started - started,
        "completion_request_seconds": time.monotonic() - completion_started,
        "finish_reason": choices[0].get("finish_reason"),
        "usage": payload.get("usage"),
        "api_key_env": api_key_env,
        "ok": True,
    }


def _artifact_records(root: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    excluded = {
        "qualification-run-manifest.json",
        "qualification-run-attestation.json",
    }
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in excluded:
            continue
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _operator.file_sha256(path),
            }
        )
    return records


def build_manifest(root: Path, state: Mapping[str, Any]) -> Dict[str, Any]:
    core = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": state.get("run_id"),
        "success": state.get("success") is True,
        "identity": state.get("identity"),
        "sources": state.get("sources"),
        "inputs": state.get("inputs"),
        "qualification_fingerprint": state.get("qualification_fingerprint"),
        "artifacts": _artifact_records(root),
        "admission": {"admitted": False},
    }
    report = {
        **core,
        "created_at_utc": _operator.utc_now(),
        "manifest_fingerprint": _operator.canonical_hash(core),
    }
    _operator.write_json(root / "qualification-run-manifest.json", report)
    return report


def verify_manifest(root: Path, *, require_success: bool = False) -> Dict[str, Any]:
    resolved = Path(root).expanduser().resolve()
    path = resolved / "qualification-run-manifest.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported qualification-run manifest schema")
    core = {
        key: value
        for key, value in report.items()
        if key not in {"created_at_utc", "manifest_fingerprint"}
    }
    if report.get("manifest_fingerprint") != _operator.canonical_hash(core):
        raise ValueError("qualification-run manifest fingerprint mismatch")
    seen: set[str] = set()
    for record in report.get("artifacts", []):
        relative = Path(str(record.get("path") or ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe artifact path in qualification-run manifest")
        normalized = relative.as_posix()
        if normalized in seen:
            raise ValueError(f"duplicate qualification artifact: {normalized}")
        seen.add(normalized)
        artifact = (resolved / relative).resolve()
        try:
            artifact.relative_to(resolved)
        except ValueError as exc:
            raise ValueError("qualification artifact escapes run directory") from exc
        if not artifact.is_file():
            raise ValueError(f"qualification artifact is missing: {normalized}")
        if artifact.stat().st_size != int(record.get("size_bytes")):
            raise ValueError(f"qualification artifact size mismatch: {normalized}")
        if _operator.file_sha256(artifact) != record.get("sha256"):
            raise ValueError(f"qualification artifact hash mismatch: {normalized}")
    if require_success and report.get("success") is not True:
        raise ValueError("qualification run did not complete successfully")
    return {
        "valid": True,
        "run_id": report.get("run_id"),
        "success": report.get("success") is True,
        "loadout_fingerprint": (report.get("identity") or {}).get(
            "loadout_fingerprint"
        ),
        "qualification_fingerprint": report.get("qualification_fingerprint"),
        "manifest_fingerprint": report.get("manifest_fingerprint"),
        "artifact_count": len(report.get("artifacts", [])),
        "admission": {"admitted": False},
    }


def _phase(
    state: Dict[str, Any],
    name: str,
    command: Sequence[str],
    root: Path,
    timeout_seconds: int,
) -> bool:
    result = _operator.run_logged(
        command,
        root / "logs" / f"{name}.log",
        timeout_seconds=timeout_seconds,
    )
    state.setdefault("phases", {})[name] = result
    _operator.write_json(root / "qualification-state.json", state)
    return result["returncode"] == 0


def _module(module: str, *args: str) -> List[str]:
    return [sys.executable, "-m", module, *args]


def run_qualification(args: argparse.Namespace) -> int:
    workspace = _workspace(args.workspace)
    current_run = validate_run_id(args.run_id or _operator.run_id())
    loadout_path = _regular_file(args.loadout, "loadout")
    inventory_path = _regular_file(args.inventory_csv, "inventory")
    cases_path = _regular_file(args.cases_file, "throughput cases")
    model_path = _regular_file(args.model_artifact, "model artifact")
    loadout_raw = load_json(loadout_path)
    if not isinstance(loadout_raw, Mapping):
        raise ValueError("loadout must be a JSON object")
    loadout = validate_manifest(
        loadout_raw,
        require_fingerprint=bool(loadout_raw.get("loadout_fingerprint")),
    )
    endpoint = require_loopback_endpoint(args.endpoint)
    inventory = inventory_identity(inventory_path, loadout, endpoint)
    if _operator.file_sha256(model_path) != loadout["model"]["content_sha256"]:
        raise ValueError("model artifact SHA-256 does not match exact loadout")

    lms_source = source_snapshot(
        args.lms_repo,
        label="LMS",
        expected_branch=args.lms_branch,
        expected_commit=args.lms_commit,
        require_contains=Path(__file__),
    )
    hermes_source = source_snapshot(
        args.hermes_repo,
        label="Hermes",
        expected_branch=args.hermes_branch,
        expected_commit=args.hermes_commit,
        required_file="run_agent.py",
    )
    preflight_probe = endpoint_probe(
        endpoint,
        loadout["model"]["id"],
        api_key_env=args.api_key_env,
        timeout_seconds=args.endpoint_timeout_seconds,
    )

    lock_workspace = workspace / ".qualification-locks"
    lock_workspace.mkdir(parents=True, exist_ok=True)
    lock, recovered = _operator.acquire_lock(
        lock_workspace,
        current_run=current_run,
        config_sha256=loadout["loadout_fingerprint"],
        recover_stale=args.recover_stale_lock,
    )
    root = workspace / current_run
    try:
        root.mkdir(parents=False, exist_ok=False)
    except BaseException:
        _operator.release_lock(lock)
        raise

    copied_loadout = root / "inputs" / "loadout.json"
    copied_inventory = root / "inputs" / "inventory.csv"
    copied_cases = root / "inputs" / "throughput-cases.json"
    copied_loadout.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(loadout_path, copied_loadout)
    shutil.copyfile(inventory_path, copied_inventory)
    shutil.copyfile(cases_path, copied_cases)

    state: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": current_run,
        "started_at_utc": _operator.utc_now(),
        "success": False,
        "identity": {
            "node_id": loadout["node_id"],
            "candidate_id": loadout["candidate_id"],
            "model_id": loadout["model"]["id"],
            "model_content_sha256": loadout["model"]["content_sha256"],
            "loadout_fingerprint": loadout["loadout_fingerprint"],
            "endpoint": endpoint,
            "loopback_only": True,
        },
        "sources": {"lms": lms_source, "hermes": hermes_source},
        "inputs": {
            "loadout": {
                "path": str(copied_loadout),
                "sha256": _operator.file_sha256(copied_loadout),
            },
            "inventory": {
                "path": str(copied_inventory),
                "sha256": _operator.file_sha256(copied_inventory),
                "identity": inventory,
            },
            "throughput_cases": {
                "path": str(copied_cases),
                "sha256": _operator.file_sha256(copied_cases),
            },
            "model_artifact": {
                "path": str(model_path),
                "sha256": _operator.file_sha256(model_path),
                "size_bytes": model_path.stat().st_size,
            },
        },
        "preflight_probe": preflight_probe,
        "recovered_lock": str(recovered) if recovered else None,
        "package_version": __version__,
        "phases": {},
        "admission": {"admitted": False},
    }
    state_path = root / "qualification-state.json"
    _operator.write_json(state_path, state)
    outcome = 1
    try:
        throughput_dir = root / "throughput"
        throughput_sidecars = root / "throughput-sidecars"
        throughput_command = _module(
            "lms_agent_bench.benchmark_entrypoint",
            "--inventory-csv",
            str(copied_inventory),
            "--cases-file",
            str(copied_cases),
            "--output-dir",
            str(throughput_dir),
            "--sidecar-dir",
            str(throughput_sidecars),
            "--trials",
            str(args.throughput_trials),
            "--min-valid-trials",
            str(args.throughput_trials),
            "--max-trial-attempts",
            str(args.max_trial_attempts),
            "--warmup-runs",
            str(args.warmup_runs),
            "--repeats",
            "1",
            "--timeout",
            str(args.request_timeout_seconds),
            "--preflight-timeout",
            str(args.endpoint_timeout_seconds),
            "--max-context-tokens",
            str(loadout["context"]["configured_tokens"]),
            "--api-key-env",
            args.api_key_env,
        )
        if not _phase(
            state,
            "throughput",
            throughput_command,
            root,
            args.throughput_phase_timeout_seconds,
        ):
            state["failure_stage"] = "throughput"
            return outcome

        base_report = root / "hermes-base.json"
        context_report = root / "hermes-context.json"
        suite_root = Path(__file__).resolve().parent / "benchmarks"
        common_hermes = [
            "--loadout",
            str(copied_loadout),
            "--hermes-repo",
            str(Path(args.hermes_repo).expanduser().resolve()),
            "--endpoint",
            endpoint,
            "--api-key",
            "local-benchmark",
            "--trials",
            str(args.hermes_trials),
            "--timeout-seconds",
            str(args.hermes_trial_timeout_seconds),
        ]
        base_command = _module(
            "lms_agent_bench.hermes_agent_bench",
            "run",
            *common_hermes,
            "--suite",
            str(suite_root / "hermes_agent_suite.v1.json"),
            "--workspace",
            str(root / "hermes-base-work"),
            "--run-id",
            "base",
            "--out",
            str(base_report),
        )
        if not _phase(
            state,
            "hermes-base",
            base_command,
            root,
            args.hermes_phase_timeout_seconds,
        ):
            state["failure_stage"] = "hermes-base"
            return outcome

        context_command = _module(
            "lms_agent_bench.hermes_agent_bench",
            "run",
            *common_hermes,
            "--suite",
            str(suite_root / "hermes_agent_context_suite.v1.json"),
            "--workspace",
            str(root / "hermes-context-work"),
            "--run-id",
            "context",
            "--out",
            str(context_report),
        )
        if not _phase(
            state,
            "hermes-context",
            context_command,
            root,
            args.hermes_phase_timeout_seconds,
        ):
            state["failure_stage"] = "hermes-context"
            return outcome

        reliability = throughput_dir / "reliability.json"
        throughput_evidence = root / "throughput-evidence.json"
        bind_command = _module(
            "lms_agent_bench.loadout_qualification",
            "bind-throughput",
            "--loadout",
            str(copied_loadout),
            "--reliability",
            str(reliability),
            "--out",
            str(throughput_evidence),
        )
        if not _phase(state, "bind-throughput", bind_command, root, 300):
            state["failure_stage"] = "bind-throughput"
            return outcome

        qualification_path = root / "loadout-qualification.json"
        qualification_command = _module(
            "lms_agent_bench.loadout_qualification",
            "qualify",
            "--loadout",
            str(copied_loadout),
            "--throughput",
            str(throughput_evidence),
            "--base-hermes",
            str(base_report),
            "--context-hermes",
            str(context_report),
            "--out",
            str(qualification_path),
        )
        if not _phase(state, "qualify", qualification_command, root, 300):
            state["failure_stage"] = "qualify"
            return outcome

        qualification_raw = load_json(qualification_path)
        if not isinstance(qualification_raw, Mapping):
            raise ValueError("qualification output is not a JSON object")
        verified = verify_qualification(qualification_raw, loadout)
        state["qualification_fingerprint"] = verified["fingerprint"]

        postflight_probe = endpoint_probe(
            endpoint,
            loadout["model"]["id"],
            api_key_env=args.api_key_env,
            timeout_seconds=args.endpoint_timeout_seconds,
        )
        post_lms = source_snapshot(
            args.lms_repo,
            label="LMS",
            expected_branch=args.lms_branch,
            expected_commit=args.lms_commit,
            require_contains=Path(__file__),
        )
        post_hermes = source_snapshot(
            args.hermes_repo,
            label="Hermes",
            expected_branch=args.hermes_branch,
            expected_commit=args.hermes_commit,
            required_file="run_agent.py",
        )
        if post_lms["source_fingerprint"] != lms_source["source_fingerprint"]:
            raise ValueError("LMS source changed during qualification")
        if post_hermes["source_fingerprint"] != hermes_source["source_fingerprint"]:
            raise ValueError("Hermes source changed during qualification")
        if _operator.file_sha256(model_path) != loadout["model"]["content_sha256"]:
            raise ValueError("model artifact changed during qualification")
        state["postflight_probe"] = postflight_probe
        state["postflight_sources"] = {"lms": post_lms, "hermes": post_hermes}
        state["success"] = True
        outcome = 0
        return outcome
    except KeyboardInterrupt:
        state["failure_stage"] = "interrupted"
        state["error"] = "qualification operator interrupted"
        outcome = 130
        return outcome
    except BaseException as exc:
        state["failure_stage"] = state.get("failure_stage") or "operator-exception"
        state["error_type"] = type(exc).__name__
        state["error"] = str(exc)
        outcome = 1
        return outcome
    finally:
        state["finished_at_utc"] = _operator.utc_now()
        _operator.write_json(state_path, state)
        try:
            build_manifest(root, state)
        finally:
            _operator.release_lock(lock)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lms-loadout-qualification-run",
        description="Run throughput and both Hermes suites for one approved exact loadout",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--loadout", type=Path, required=True)
    run.add_argument("--inventory-csv", type=Path, required=True)
    run.add_argument("--cases-file", type=Path, required=True)
    run.add_argument("--model-artifact", type=Path, required=True)
    run.add_argument("--endpoint", required=True)
    run.add_argument("--api-key-env", default="LMSTUDIO_API_KEY")
    run.add_argument("--lms-repo", type=Path, required=True)
    run.add_argument("--lms-branch", required=True)
    run.add_argument("--lms-commit", required=True)
    run.add_argument("--hermes-repo", type=Path, required=True)
    run.add_argument("--hermes-branch", required=True)
    run.add_argument("--hermes-commit", required=True)
    run.add_argument("--workspace", type=Path, required=True)
    run.add_argument("--run-id")
    run.add_argument("--recover-stale-lock", action="store_true")
    run.add_argument("--throughput-trials", type=int, default=3)
    run.add_argument("--max-trial-attempts", type=int, default=5)
    run.add_argument("--warmup-runs", type=int, default=3)
    run.add_argument("--request-timeout-seconds", type=float, default=900.0)
    run.add_argument("--endpoint-timeout-seconds", type=float, default=120.0)
    run.add_argument("--throughput-phase-timeout-seconds", type=int, default=21600)
    run.add_argument("--hermes-trials", type=int, default=3)
    run.add_argument("--hermes-trial-timeout-seconds", type=float, default=600.0)
    run.add_argument("--hermes-phase-timeout-seconds", type=int, default=21600)
    verify = commands.add_parser("verify")
    verify.add_argument("--run-dir", type=Path, required=True)
    verify.add_argument("--require-success", action="store_true")
    return parser


def _validate_positive(args: argparse.Namespace) -> None:
    if args.command != "run":
        return
    for name in (
        "throughput_trials",
        "max_trial_attempts",
        "warmup_runs",
        "request_timeout_seconds",
        "endpoint_timeout_seconds",
        "throughput_phase_timeout_seconds",
        "hermes_trials",
        "hermes_trial_timeout_seconds",
        "hermes_phase_timeout_seconds",
    ):
        if float(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.max_trial_attempts < args.throughput_trials:
        raise ValueError("max_trial_attempts cannot be less than throughput_trials")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _validate_positive(args)
        if args.command == "run":
            return run_qualification(args)
        report = verify_manifest(args.run_dir, require_success=args.require_success)
    except (
        OSError,
        ValueError,
        RuntimeError,
        json.JSONDecodeError,
        csv.Error,
        requests.RequestException,
    ) as exc:
        print(f"qualification operator failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
