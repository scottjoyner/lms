"""Deterministic, fail-closed operator workflow for physical fleet evidence.

This command deliberately exposes a small surface. It validates one complete
configuration, probes every runnable node with a fixed SSH script, renders the
remote scripts, runs observation, collects artifacts, and executes the release
gate. It never invents candidate IDs, admits runtimes, or modifies routing.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from lms_agent_bench import fleet_rollout_entrypoint
from lms_agent_bench.fleet_coverage import validate_rollout_coverage

SCHEMA_VERSION = "fleet_operator_run.v1"
SSH_OPTIONS = (
    "BatchMode=yes",
    "ConnectTimeout=10",
    "ConnectionAttempts=1",
    "ServerAliveInterval=15",
    "ServerAliveCountMax=2",
    "StrictHostKeyChecking=accept-new",
    "LogLevel=ERROR",
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def run_id() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_operator_config(config_path: str, env_file: str) -> Dict[str, Any]:
    config = fleet_rollout_entrypoint.load_rollout_config(config_path, env_file)
    coverage = validate_rollout_coverage(config, config_path)
    if not coverage.get("ready"):
        raise ValueError(
            "; ".join(coverage.get("errors") or ["fleet coverage is not ready"])
        )
    return {"config": config, "coverage": coverage}


def ssh_command(target: str) -> List[str]:
    command = ["ssh"]
    for option in SSH_OPTIONS:
        command.extend(["-o", option])
    command.extend([target, "bash", "-s"])
    return command


def preflight_script(node: Mapping[str, Any], update_code: bool) -> str:
    values = {
        "LMS_PREFLIGHT_NODE": str(node["node_id"]),
        "LMS_PREFLIGHT_REPO": str(node["repo_dir"]),
        "LMS_PREFLIGHT_BRANCH": str(node["branch"]),
        "LMS_PREFLIGHT_COMMIT": str(node.get("expected_commit") or ""),
        "LMS_PREFLIGHT_PYTHON": str(node.get("python") or "python3"),
        "LMS_PREFLIGHT_ROOTS": json.dumps(node.get("model_roots") or []),
        "LMS_PREFLIGHT_ALLOW_UPDATE": "1" if update_code else "0",
    }
    exports = "\n".join(
        f"export {key}={shlex.quote(value)}" for key, value in values.items()
    )
    return exports + r'''
set -euo pipefail
PYTHON_BIN="$LMS_PREFLIGHT_PYTHON"
if [[ "$PYTHON_BIN" == */* ]]; then
  test -x "$PYTHON_BIN"
else
  command -v "$PYTHON_BIN" >/dev/null
fi
"$PYTHON_BIN" - <<'PY'
import importlib.util
import json
import os
import pathlib
import platform
import subprocess
import sys

node = os.environ["LMS_PREFLIGHT_NODE"]
repo = pathlib.Path(os.path.expanduser(os.environ["LMS_PREFLIGHT_REPO"]))
expected_branch = os.environ["LMS_PREFLIGHT_BRANCH"]
expected_commit = os.environ["LMS_PREFLIGHT_COMMIT"].lower()
allow_update = os.environ["LMS_PREFLIGHT_ALLOW_UPDATE"] == "1"
roots = [pathlib.Path(os.path.expanduser(item)) for item in json.loads(os.environ["LMS_PREFLIGHT_ROOTS"])]

def git(*args):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()

if not (repo / ".git").is_dir():
    raise SystemExit(f"not a git checkout: {repo}")
if importlib.util.find_spec("requests") is None:
    raise SystemExit("remote Python cannot import requests")
missing_roots = [str(path) for path in roots if not path.is_dir()]
if missing_roots:
    raise SystemExit("missing model roots: " + ", ".join(missing_roots))
status = git("status", "--porcelain", "--untracked-files=all")
if status:
    raise SystemExit("remote checkout is not completely clean")
branch = git("branch", "--show-current")
commit = git("rev-parse", "HEAD").lower()
if not allow_update:
    if branch != expected_branch:
        raise SystemExit(f"expected branch {expected_branch}, found {branch or 'detached'}")
    if commit != expected_commit:
        raise SystemExit(f"expected commit {expected_commit}, found {commit}")
print(json.dumps({
    "node_id": node,
    "hostname": platform.node(),
    "platform": platform.platform(),
    "python": sys.executable,
    "repo": str(repo),
    "branch": branch,
    "commit": commit,
    "model_roots": [str(path) for path in roots],
    "clean": True,
    "update_code_requested": allow_update,
}, sort_keys=True))
PY
'''


def preflight_node(node: Mapping[str, Any], update_code: bool) -> Dict[str, Any]:
    started = utc_now()
    try:
        proc = subprocess.run(
            ssh_command(str(node["ssh_target"])),
            input=preflight_script(node, update_code),
            text=True,
            capture_output=True,
            timeout=45,
            check=False,
        )
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        proc = None
        timed_out = True
        stdout = str(exc.stdout or "")
        stderr = str(exc.stderr or "") + "\nSSH preflight timed out after 45 seconds"
        return {
            "node_id": node["node_id"],
            "ssh_target": node["ssh_target"],
            "started_at_utc": started,
            "finished_at_utc": utc_now(),
            "returncode": 124,
            "timed_out": True,
            "ok": False,
            "stdout": stdout.strip(),
            "stderr": stderr.strip(),
        }
    assert proc is not None
    detail: Optional[Dict[str, Any]] = None
    if proc.returncode == 0:
        lines = [line for line in proc.stdout.splitlines() if line.strip()]
        if lines:
            try:
                parsed = json.loads(lines[-1])
                if isinstance(parsed, dict):
                    detail = parsed
            except json.JSONDecodeError:
                detail = None
    return {
        "node_id": node["node_id"],
        "ssh_target": node["ssh_target"],
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "returncode": proc.returncode,
        "timed_out": timed_out,
        "ok": proc.returncode == 0 and detail is not None,
        "detail": detail,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def run_preflight(
    config: Mapping[str, Any], update_code: bool
) -> List[Dict[str, Any]]:
    results = []
    for node in config["nodes"]:
        result = preflight_node(node, update_code)
        results.append(result)
    return results


def run_logged(command: Sequence[str], log_path: Path) -> int:
    proc = subprocess.run(
        list(command),
        text=True,
        capture_output=True,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "$ " + " ".join(shlex.quote(item) for item in command)
        + "\n\n[stdout]\n"
        + proc.stdout
        + "\n[stderr]\n"
        + proc.stderr,
        encoding="utf-8",
    )
    return proc.returncode


def module_command(module: str, *args: str) -> List[str]:
    return [sys.executable, "-m", module, *args]


def acquire_lock(workspace: Path) -> Path:
    lock = workspace / ".fleet-operator.lock"
    try:
        lock.mkdir(parents=False)
    except FileExistsError as exc:
        raise RuntimeError(f"fleet operator lock already exists: {lock}") from exc
    (lock / "owner.json").write_text(
        json.dumps({"pid": os.getpid(), "started_at_utc": utc_now()}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return lock


def release_lock(lock: Path) -> None:
    try:
        for child in lock.iterdir():
            child.unlink()
        lock.rmdir()
    except OSError:
        pass


def preflight_command(args: argparse.Namespace) -> int:
    loaded = load_operator_config(args.config, args.env_file)
    workspace = Path(args.workspace).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    results = run_preflight(loaded["config"], args.update_code)
    report = {
        "schema_version": SCHEMA_VERSION,
        "phase": "preflight",
        "created_at_utc": utc_now(),
        "config": str(Path(args.config).resolve()),
        "env_file": str(Path(args.env_file).resolve()),
        "coverage": loaded["coverage"],
        "results": results,
        "ok": all(item["ok"] for item in results),
        "admission": {"admitted": False},
    }
    write_json(workspace / "preflight.json", report)
    return 0 if report["ok"] else 1


def observe_command(args: argparse.Namespace) -> int:
    loaded = load_operator_config(args.config, args.env_file)
    workspace = Path(args.workspace).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    lock = acquire_lock(workspace)
    current_run = args.run_id or run_id()
    root = workspace / current_run
    root.mkdir(parents=True, exist_ok=False)
    state: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "phase": "observe",
        "run_id": current_run,
        "started_at_utc": utc_now(),
        "config": str(Path(args.config).resolve()),
        "env_file": str(Path(args.env_file).resolve()),
        "coverage": loaded["coverage"],
        "node_ids": [item["node_id"] for item in loaded["config"]["nodes"]],
        "update_code": args.update_code,
        "success": False,
        "admission": {"admitted": False},
    }
    state_path = root / "operator-state.json"
    write_json(state_path, state)
    try:
        preflight = run_preflight(loaded["config"], args.update_code)
        state["preflight"] = preflight
        write_json(root / "preflight.json", {
            "schema_version": SCHEMA_VERSION,
            "phase": "preflight",
            "coverage": loaded["coverage"],
            "results": preflight,
            "ok": all(item["ok"] for item in preflight),
            "admission": {"admitted": False},
        })
        if not all(item["ok"] for item in preflight):
            state["failure_stage"] = "preflight"
            state["finished_at_utc"] = utc_now()
            write_json(state_path, state)
            return 1

        common = [
            "--config", args.config,
            "--env-file", args.env_file,
            "--all-nodes",
        ]
        render_args = [
            "render", *common,
            "--run-id", current_run,
            "--output-dir", str(root / "render"),
        ]
        if args.update_code:
            render_args.append("--update-code")
        rc = run_logged(
            module_command("lms_agent_bench.fleet_rollout_complete", *render_args),
            root / "logs" / "render.log",
        )
        state["render_returncode"] = rc
        if rc != 0:
            state["failure_stage"] = "render"
            state["finished_at_utc"] = utc_now()
            write_json(state_path, state)
            return rc

        rollout_args = [
            "run", *common,
            "--run-id", current_run,
            "--continue-on-error",
            "--output-dir", str(root / "observe"),
        ]
        if args.update_code:
            rollout_args.append("--update-code")
        rc = run_logged(
            module_command("lms_agent_bench.fleet_rollout_complete", *rollout_args),
            root / "logs" / "observe.log",
        )
        state["rollout_returncode"] = rc
        if rc != 0:
            state["failure_stage"] = "rollout"
            state["finished_at_utc"] = utc_now()
            write_json(state_path, state)
            return rc

        gate_args = [
            "--mode", "observe",
            "--rollout-results", str(root / "observe" / "rollout_results.json"),
            "--out", str(root / "observe" / "release-gate.json"),
        ]
        for node_id in state["node_ids"]:
            gate_args.extend(["--required-node", node_id])
        rc = run_logged(
            module_command("lms_agent_bench.fleet_gate_entrypoint", *gate_args),
            root / "logs" / "gate.log",
        )
        state["gate_returncode"] = rc
        state["success"] = rc == 0
        if rc != 0:
            state["failure_stage"] = "gate"
        state["finished_at_utc"] = utc_now()
        write_json(state_path, state)
        return rc
    finally:
        release_lock(lock)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run deterministic fail-closed physical fleet workflows"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "observe"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--config", required=True)
        cmd.add_argument("--env-file", required=True)
        cmd.add_argument("--workspace", required=True)
        cmd.add_argument("--update-code", action="store_true")
        if name == "observe":
            cmd.add_argument("--run-id", default=None)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "preflight":
            return preflight_command(args)
        return observe_command(args)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"fleet operator failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
