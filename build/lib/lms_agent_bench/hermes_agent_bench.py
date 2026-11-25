#!/usr/bin/env python3
"""Run real Hermes agent loops against an exact model loadout and MCP fixture."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from lms_agent_bench.hermes_agent_common import (
    SCHEMA_VERSION,
    canonical_hash,
    load_json,
    require_loopback_endpoint,
    suite_default_path,
    utc_now_iso,
    validate_suite,
    write_json,
)
from lms_agent_bench.hermes_agent_scoring import aggregate_trials, evaluate_gate, evaluate_trial, verify_report
from lms_agent_bench.model_loadout import validate_manifest


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): json_safe(child) for key, child in value.items() if not str(key).startswith("_db_")}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(child) for child in value]
    if hasattr(value, "model_dump"):
        try:
            return json_safe(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "to_dict"):
        try:
            return json_safe(value.to_dict())
        except Exception:
            pass
    return str(value)


def context_pressure_text(case: Mapping[str, Any], loadout: Mapping[str, Any]) -> str:
    ratio = float(case.get("context_pressure_ratio") or 0.0)
    explicit = case.get("context_pressure_tokens")
    if explicit is not None:
        target = int(explicit)
    else:
        target = int(int(loadout["context"]["configured_tokens"]) * ratio)
    if target <= 0:
        return ""
    configured = int(loadout["context"]["configured_tokens"])
    target = min(target, max(0, configured - 2048))
    if target <= 0:
        return ""
    control = str(case.get("context_control_code") or "LMS-CONTEXT-5179")
    sentence = (
        "Synthetic operational note: component alpha remains nominal; "
        "component beta has no benchmark relevance; preserve chronological tool state. "
    )
    approximate_chars = target * 4
    chunks: List[str] = []
    while sum(len(item) for item in chunks) < approximate_chars:
        index = len(chunks)
        if index == 3:
            chunks.append(f"CONTROL FACT: the required benchmark control code is {control}. ")
        else:
            chunks.append(f"Note {index:05d}. {sentence}")
    return "\n\n<synthetic_context>\n" + "".join(chunks)[:approximate_chars] + "\n</synthetic_context>\n"


def seed_workspace(workspace: Path, files: Mapping[str, Any]) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    for relative, content in files.items():
        path = Path(str(relative))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe workspace seed path: {relative}")
        destination = workspace / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(str(content), encoding="utf-8")


def trial_command(args: argparse.Namespace, *, case_path: Path, loadout_path: Path, result_path: Path, home: Path, workspace: Path, state_dir: Path) -> List[str]:
    return [
        args.hermes_python,
        str(Path(__file__).resolve()),
        "_trial",
        "--hermes-repo",
        str(Path(args.hermes_repo).resolve()),
        "--case",
        str(case_path),
        "--loadout",
        str(loadout_path),
        "--endpoint",
        args.endpoint,
        "--api-key",
        args.api_key,
        "--result",
        str(result_path),
        "--hermes-home",
        str(home),
        "--workspace",
        str(workspace),
        "--state-dir",
        str(state_dir),
    ]


def run_one_trial(args: argparse.Namespace, case: Mapping[str, Any], loadout: Mapping[str, Any], trial_index: int, root: Path, prohibited_tools: set[str]) -> Dict[str, Any]:
    case_root = root / str(case["case_key"]) / f"trial-{trial_index:02d}"
    workspace = case_root / "workspace"
    state_dir = case_root / "fixture-state"
    home = case_root / "hermes-home"
    result_path = case_root / "agent-result.json"
    case_path = case_root / "case.json"
    loadout_path = case_root / "loadout.json"
    case_root.mkdir(parents=True, exist_ok=False)
    seed_workspace(workspace, case.get("workspace", {}))
    write_json(case_path, dict(case))
    write_json(loadout_path, dict(loadout))
    command = trial_command(
        args,
        case_path=case_path,
        loadout_path=loadout_path,
        result_path=result_path,
        home=home,
        workspace=workspace,
        state_dir=state_dir,
    )
    started = time.monotonic()
    timed_out = False
    try:
        process = subprocess.run(
            command,
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=float(args.timeout_seconds),
            check=False,
            env={**os.environ, "NO_COLOR": "1", "TERM": "dumb"},
        )
        returncode = process.returncode
        stdout = process.stdout
        stderr = process.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = 124
        stdout = str(exc.stdout or "")
        stderr = str(exc.stderr or "") + f"\ntrial timeout after {args.timeout_seconds}s"
    wall = time.monotonic() - started
    try:
        process_result = load_json(result_path)
        if not isinstance(process_result, Mapping):
            process_result = {"ok": False, "error": "trial result is not an object"}
    except (OSError, json.JSONDecodeError):
        process_result = {"ok": False, "error": "trial produced no valid result artifact"}
    process_result = dict(process_result)
    process_result.setdefault("wall_seconds", wall)
    fixture_calls = []
    calls_path = state_dir / "calls.jsonl"
    if calls_path.is_file():
        for line in calls_path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                fixture_calls.append(item)
    trial = evaluate_trial(
        case=case,
        trial_index=trial_index,
        process_result=process_result,
        process_returncode=returncode,
        timed_out=timed_out,
        stdout=stdout,
        stderr=stderr,
        fixture_calls=fixture_calls,
        workspace=workspace,
        prohibited_tools=prohibited_tools,
    )
    write_json(case_root / "trial-report.json", trial)
    return trial


def run_benchmark(args: argparse.Namespace) -> int:
    raw_loadout = load_json(Path(args.loadout))
    if not isinstance(raw_loadout, Mapping):
        raise ValueError("loadout must be a JSON object")
    loadout = validate_manifest(raw_loadout, require_fingerprint=bool(raw_loadout.get("loadout_fingerprint")))
    endpoint = require_loopback_endpoint(args.endpoint)
    suite_raw = load_json(Path(args.suite))
    if not isinstance(suite_raw, Mapping):
        raise ValueError("suite must be a JSON object")
    suite = validate_suite(suite_raw)
    trials_per_case = int(args.trials or suite["minimum_valid_trials"])
    if trials_per_case < suite["minimum_valid_trials"]:
        raise ValueError("requested trials are below the suite minimum")
    root = Path(args.workspace).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_root = root / (args.run_id or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    run_root.mkdir(parents=True, exist_ok=False)
    selected_cases = suite["cases"]
    if args.case:
        requested = set(args.case)
        selected_cases = [item for item in selected_cases if item["case_key"] in requested]
        missing = sorted(requested - {item["case_key"] for item in selected_cases})
        if missing:
            raise ValueError("unknown case keys: " + ", ".join(missing))
    prohibited = {str(item) for item in suite.get("prohibited_tools", [])}
    all_trials: List[Dict[str, Any]] = []
    if not args.dry_run:
        for case in selected_cases:
            for trial_index in range(1, trials_per_case + 1):
                all_trials.append(run_one_trial(args, case, loadout, trial_index, run_root, prohibited))
    aggregate = aggregate_trials({**suite, "cases": selected_cases}, all_trials)
    gate = evaluate_gate({**suite, "cases": selected_cases}, aggregate)
    if args.dry_run:
        gate = {**gate, "passed": False, "intelligence_qualified": False, "failures": ["dry run contains no physical agent evidence"]}
    identity = {
        "node_id": loadout["node_id"],
        "candidate_id": loadout["candidate_id"],
        "model_id": loadout["model"]["id"],
        "model_content_sha256": loadout["model"]["content_sha256"],
        "loadout_fingerprint": loadout["loadout_fingerprint"],
        "architecture_kind": loadout["architecture"]["kind"],
        "total_parameter_count": loadout["architecture"]["total_parameter_count"],
        "active_parameter_count_per_token": loadout["architecture"].get("active_parameter_count_per_token"),
        "weight_quantization": loadout["weight_quantization"],
        "configured_context_tokens": loadout["context"]["configured_tokens"],
        "kv_cache": loadout["kv_cache"],
        "parallel_slots": loadout["concurrency"]["parallel_slots"],
        "runtime": loadout["runtime"],
        "endpoint": endpoint,
        "loopback_only": True,
    }
    core = {
        "identity": identity,
        "suite_id": suite["suite_id"],
        "suite_fingerprint": canonical_hash(suite),
        "trials_per_case": trials_per_case,
        "trials": all_trials,
        "aggregate": aggregate,
        "gate": gate,
        "dry_run": bool(args.dry_run),
        "admission": {"admitted": False},
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "hermes_agent_intelligence_benchmark",
        "created_at_utc": utc_now_iso(),
        "run_id": run_root.name,
        "loadout": loadout,
        **core,
        "benchmark_fingerprint": canonical_hash(core),
    }
    write_json(Path(args.out), report)
    write_json(run_root / "benchmark-report.json", report)
    return 0 if gate["passed"] else 1


def write_trial_config(home: Path, state_dir: Path, workspace: Path, scenario: str) -> None:
    home.mkdir(parents=True, exist_ok=True)
    fixture = Path(__file__).resolve().with_name("hermes_mcp_fixture.py")
    config = {
        "mcp_servers": {
            "lms-benchmark": {
                "command": sys.executable,
                "args": [
                    str(fixture),
                    "--state-dir",
                    str(state_dir),
                    "--workspace",
                    str(workspace),
                    "--scenario",
                    scenario,
                ],
                "env": {},
                "timeout": 60,
                "connect_timeout": 30,
                "supports_parallel_tool_calls": False,
            }
        }
    }
    (home / "config.yaml").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def trial_process(args: argparse.Namespace) -> int:
    started = time.monotonic()
    result_artifact: Dict[str, Any]
    try:
        case = load_json(Path(args.case))
        loadout = validate_manifest(load_json(Path(args.loadout)))
        if not isinstance(case, Mapping):
            raise ValueError("case must be an object")
        home = Path(args.hermes_home).resolve()
        workspace = Path(args.workspace).resolve()
        state_dir = Path(args.state_dir).resolve()
        write_trial_config(home, state_dir, workspace, str(case.get("scenario") or "default"))
        os.environ["HERMES_HOME"] = str(home)
        os.environ["OPENAI_API_KEY"] = args.api_key
        os.environ["NO_COLOR"] = "1"
        os.environ["TERM"] = "dumb"
        hermes_repo = Path(args.hermes_repo).resolve()
        if not (hermes_repo / "run_agent.py").is_file():
            raise ValueError("Hermes repository does not contain run_agent.py")
        sys.path.insert(0, str(hermes_repo))
        previous_cwd = Path.cwd()
        os.chdir(workspace)
        try:
            from run_agent import AIAgent

            prompt = str(case.get("prompt") or "") + context_pressure_text(case, loadout)
            control = case.get("context_control_code")
            if control:
                prompt += f"\nReturn the context control code {control} in your final response in addition to completing the task."
            agent = AIAgent(
                model=loadout["model"]["id"],
                base_url=require_loopback_endpoint(args.endpoint),
                api_key=args.api_key,
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                save_trajectories=False,
                max_iterations=int(case.get("max_iterations") or 20),
                enabled_toolsets=["mcp-lms-benchmark"],
            )
            raw = agent.run_conversation(
                user_message=prompt,
                system_message=str(case.get("system") or ""),
                task_id=f"lms-benchmark-{case.get('case_key')}",
            )
        finally:
            os.chdir(previous_cwd)
        if not isinstance(raw, Mapping):
            raise RuntimeError(f"Hermes returned {type(raw).__name__}, expected dict")
        result_artifact = {
            "ok": True,
            "wall_seconds": time.monotonic() - started,
            "result": json_safe(raw),
        }
        returncode = 0
    except BaseException as exc:
        result_artifact = {
            "ok": False,
            "wall_seconds": time.monotonic() - started,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(limit=30),
        }
        returncode = 1
    write_json(Path(args.result), result_artifact)
    return returncode


def gate_report(args: argparse.Namespace) -> int:
    report = load_json(Path(args.report))
    if not isinstance(report, Mapping):
        raise ValueError("report must be a JSON object")
    gate = verify_report(report, args)
    write_json(Path(args.out), gate)
    return 0 if gate["passed"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark real Hermes agent loops for one exact model loadout")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--loadout", required=True)
    run.add_argument("--hermes-repo", required=True)
    run.add_argument("--hermes-python", default=sys.executable)
    run.add_argument("--endpoint", required=True)
    run.add_argument("--api-key", default="local-benchmark")
    run.add_argument("--suite", default=str(suite_default_path()))
    run.add_argument("--trials", type=int, default=None)
    run.add_argument("--case", action="append", default=[])
    run.add_argument("--timeout-seconds", type=float, default=600.0)
    run.add_argument("--workspace", required=True)
    run.add_argument("--run-id", default=None)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--out", required=True)
    gate = sub.add_parser("gate")
    gate.add_argument("--report", required=True)
    gate.add_argument("--node-id")
    gate.add_argument("--candidate-id")
    gate.add_argument("--model")
    gate.add_argument("--model-content-sha256")
    gate.add_argument("--out", required=True)
    hidden = sub.add_parser("_trial")
    hidden.add_argument("--hermes-repo", required=True)
    hidden.add_argument("--case", required=True)
    hidden.add_argument("--loadout", required=True)
    hidden.add_argument("--endpoint", required=True)
    hidden.add_argument("--api-key", required=True)
    hidden.add_argument("--result", required=True)
    hidden.add_argument("--hermes-home", required=True)
    hidden.add_argument("--workspace", required=True)
    hidden.add_argument("--state-dir", required=True)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            return run_benchmark(args)
        if args.command == "gate":
            return gate_report(args)
        return trial_process(args)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"Hermes benchmark failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
