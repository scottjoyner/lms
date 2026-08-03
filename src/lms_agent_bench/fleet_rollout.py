#!/usr/bin/env python3
"""Render or execute guarded LMS fleet rollout scripts over SSH.

The default path is read-only discovery plus dry-run command rendering. Real
candidate execution requires an exact ``NODE_ID=CANDIDATE_ID`` allow-list.
Persistent services, model registration, admission, and routing are never
modified by this tool.
"""
from __future__ import annotations

import argparse
import datetime as dt
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from lms_agent_bench.fleet_loadout import canonical_hash, load_json, write_json

SCHEMA_VERSION = "fleet_rollout.v1"
DEFAULT_SUITE_FILE = "src/lms_agent_bench/benchmarks/agent_skill_suite.v1.json"


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in value).strip("-") or "node"


def q(value: Any) -> str:
    return shlex.quote(str(value))


def validate_node(node: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    for field in ("node_id", "ssh_target", "repo_dir", "branch"):
        if not str(node.get(field, "")).strip():
            errors.append(f"node missing {field}")
    if not isinstance(node.get("model_roots"), list) or not node.get("model_roots"):
        errors.append(f"node {node.get('node_id', '<unknown>')} requires model_roots")
    contexts = node.get("contexts", [4096, 8192])
    if not isinstance(contexts, list) or not contexts or any(
        not isinstance(value, int) or value <= 0 for value in contexts
    ):
        errors.append(f"node {node.get('node_id', '<unknown>')} contexts must be positive integers")
    endpoint_map = node.get("endpoint_map", {})
    if endpoint_map is not None and not isinstance(endpoint_map, dict):
        errors.append(f"node {node.get('node_id', '<unknown>')} endpoint_map must be an object")
    return errors


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    nodes = config.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("config requires a non-empty nodes list")
    errors: List[str] = []
    seen: set[str] = set()
    for raw in nodes:
        if not isinstance(raw, Mapping):
            errors.append("every node must be an object")
            continue
        errors.extend(validate_node(raw))
        node_id = str(raw.get("node_id", ""))
        if node_id in seen:
            errors.append(f"duplicate node_id: {node_id}")
        seen.add(node_id)
    if errors:
        raise ValueError("; ".join(errors))


def node_map(config: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    validate_config(config)
    return {str(item["node_id"]): dict(item) for item in config["nodes"]}


def resolve_nodes(
    config: Mapping[str, Any], requested: Sequence[str], all_nodes: bool
) -> List[Dict[str, Any]]:
    by_id = node_map(config)
    if requested:
        unknown = sorted(set(requested) - set(by_id))
        if unknown:
            raise ValueError(f"unknown node IDs: {', '.join(unknown)}")
        return [by_id[node_id] for node_id in requested]
    if all_nodes:
        return list(by_id.values())
    raise ValueError("choose one or more --node values or pass --all-nodes")


def parse_execute_candidates(values: Sequence[str]) -> Dict[str, List[str]]:
    parsed: Dict[str, List[str]] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--execute-candidate must be NODE_ID=CANDIDATE_ID")
        node_id, candidate_id = (item.strip() for item in value.split("=", 1))
        if not node_id or not candidate_id:
            raise ValueError("--execute-candidate must be NODE_ID=CANDIDATE_ID")
        parsed.setdefault(node_id, []).append(candidate_id)
    return parsed


def script_list_flags(flag: str, values: Sequence[Any]) -> str:
    return " ".join(f"{flag} {q(value)}" for value in values)


def remote_artifact_path(node: Mapping[str, Any], run_id: str) -> str:
    root = str(node.get("artifact_root") or "~/.local/state/lms-fleet")
    return f"{root.rstrip('/')}/{run_id}/{safe_slug(str(node['node_id']))}"


def build_remote_script(
    node: Mapping[str, Any],
    run_id: str,
    execute_candidates: Sequence[str] = (),
    update_code: bool = False,
    dry_run_limit: int = 4,
) -> str:
    repo = str(node["repo_dir"])
    branch = str(node["branch"])
    python_bin = str(node.get("python") or "python3")
    artifact_path = remote_artifact_path(node, run_id)
    suite_file = str(node.get("suite_file") or DEFAULT_SUITE_FILE)
    contexts = ",".join(str(value) for value in node.get("contexts", [4096, 8192]))
    max_candidates = int(node.get("max_candidates") or 32)
    default_max_context = int(node.get("default_max_context") or max(node.get("contexts", [8192])))
    model_roots = [str(item) for item in node.get("model_roots", [])]
    endpoints = [str(item) for item in node.get("endpoints", [])]
    endpoint_map = {
        str(key): str(value) for key, value in (node.get("endpoint_map") or {}).items()
    }

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"export NODE_ID={q(node['node_id'])}",
        f"REPO_DIR={q(repo)}",
        f"EXPECTED_BRANCH={q(branch)}",
        f"PYTHON_BIN={q(python_bin)}",
        f"RAW_ARTIFACT_DIR={q(artifact_path)}",
        "ARTIFACT_DIR=$($PYTHON_BIN - \"$RAW_ARTIFACT_DIR\" <<'PY'",
        "import os, sys",
        "print(os.path.abspath(os.path.expanduser(sys.argv[1])))",
        "PY",
        ")",
        "test -d \"$REPO_DIR/.git\" || { echo \"not a git repository: $REPO_DIR\" >&2; exit 20; }",
    ]
    if update_code:
        lines.extend(
            [
                "git -C \"$REPO_DIR\" fetch --prune origin \"$EXPECTED_BRANCH\"",
                "git -C \"$REPO_DIR\" checkout \"$EXPECTED_BRANCH\"",
                "git -C \"$REPO_DIR\" pull --ff-only origin \"$EXPECTED_BRANCH\"",
            ]
        )
    else:
        lines.extend(
            [
                "CURRENT_BRANCH=$(git -C \"$REPO_DIR\" branch --show-current)",
                "test \"$CURRENT_BRANCH\" = \"$EXPECTED_BRANCH\" || { echo \"expected branch $EXPECTED_BRANCH, found ${CURRENT_BRANCH:-detached}\" >&2; exit 21; }",
            ]
        )
    lines.extend(
        [
            "mkdir -p \"$ARTIFACT_DIR\"",
            "export PYTHONPATH=\"$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}\"",
            "$PYTHON_BIN -c 'import requests' >/dev/null",
            "cd \"$REPO_DIR\"",
            (
                "$PYTHON_BIN -m lms_agent_bench.fleet_loadout discover "
                f"{script_list_flags('--endpoint', endpoints)} "
                "--out \"$ARTIFACT_DIR/machine_observation.json\""
            ).strip(),
            (
                "$PYTHON_BIN -m lms_agent_bench.fleet_models scan "
                f"{script_list_flags('--root', model_roots)} "
                "--hash-mode quick "
                f"--default-max-context {default_max_context} "
                "--out \"$ARTIFACT_DIR/model_inventory.json\""
            ),
            (
                "$PYTHON_BIN -m lms_agent_bench.fleet_loadout plan "
                "--observation \"$ARTIFACT_DIR/machine_observation.json\" "
                "--models \"$ARTIFACT_DIR/model_inventory.json\" "
                f"--contexts {q(contexts)} --max-candidates {max_candidates} "
                "--out \"$ARTIFACT_DIR/benchmark_plan.json\""
            ),
        ]
    )

    benchmark_base = (
        "$PYTHON_BIN -m lms_agent_bench.fleet_bench_plan "
        "--plan \"$ARTIFACT_DIR/benchmark_plan.json\" "
        "--output-dir \"$ARTIFACT_DIR/benchmark\" "
        f"--suite-file \"$REPO_DIR/{suite_file}\" "
    )
    for candidate_id, url in endpoint_map.items():
        benchmark_base += f"--endpoint-map {q(candidate_id + '=' + url)} "

    if execute_candidates:
        candidate_flags = script_list_flags("--candidate", execute_candidates)
        lines.extend(
            [
                f"{benchmark_base}{candidate_flags}",
                (
                    "$PYTHON_BIN -m lms_agent_bench.fleet_loadout select "
                    "--plan \"$ARTIFACT_DIR/benchmark_plan.json\" "
                    "--results-csv \"$ARTIFACT_DIR/benchmark/loadout_results.csv\" "
                    "--out \"$ARTIFACT_DIR/selected_loadout.json\""
                ),
                (
                    "$PYTHON_BIN -m lms_agent_bench.fleet_models fingerprint "
                    "--inventory \"$ARTIFACT_DIR/model_inventory.json\" "
                    "--selection \"$ARTIFACT_DIR/selected_loadout.json\" "
                    "--out \"$ARTIFACT_DIR/model_inventory.selected.json\""
                ),
            ]
        )
    else:
        lines.append(f"{benchmark_base}--all --limit {max(1, dry_run_limit)} --dry-run")

    lines.extend(
        [
            "$PYTHON_BIN - \"$ARTIFACT_DIR\" <<'PY'",
            "import hashlib, json, os, pathlib, sys",
            "root = pathlib.Path(sys.argv[1])",
            "files = []",
            "for path in sorted(p for p in root.rglob('*') if p.is_file()):",
            "    digest = hashlib.sha256(path.read_bytes()).hexdigest()",
            "    files.append({'path': str(path.relative_to(root)), 'size_bytes': path.stat().st_size, 'sha256': 'sha256:' + digest})",
            "manifest = {'schema_version': 'fleet_artifact_bundle.v1', 'node_id': os.environ.get('NODE_ID'), 'files': files}",
            "(root / 'bundle_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\\n')",
            "PY",
            "tar -C \"$ARTIFACT_DIR\" -czf \"$ARTIFACT_DIR.tar.gz\" .",
            "echo \"LMS_FLEET_ARTIFACT=$ARTIFACT_DIR.tar.gz\"",
        ]
    )
    return "\n".join(lines) + "\n"


def ssh_command(target: str, options: Sequence[str]) -> List[str]:
    command = ["ssh"]
    for option in options:
        command.extend(["-o", option])
    command.extend([target, "bash", "-s"])
    return command


def scp_command(target: str, remote_path: str, local_path: Path, options: Sequence[str]) -> List[str]:
    command = ["scp"]
    for option in options:
        command.extend(["-o", option])
    command.extend([f"{target}:{remote_path}", str(local_path)])
    return command


def execute_remote(
    node: Mapping[str, Any],
    script: str,
    run_id: str,
    collect_dir: Path,
    ssh_options: Sequence[str],
    collect: bool,
) -> Dict[str, Any]:
    started_at = utc_now_iso()
    proc = subprocess.run(
        ssh_command(str(node["ssh_target"]), ssh_options),
        input=script,
        text=True,
        capture_output=True,
        check=False,
    )
    result: Dict[str, Any] = {
        "node_id": node["node_id"],
        "ssh_target": node["ssh_target"],
        "started_at_utc": started_at,
        "finished_at_utc": utc_now_iso(),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "collected_artifact": None,
    }
    if proc.returncode == 0 and collect:
        remote_tar = remote_artifact_path(node, run_id) + ".tar.gz"
        collect_dir.mkdir(parents=True, exist_ok=True)
        local_tar = collect_dir / f"{safe_slug(str(node['node_id']))}.tar.gz"
        scp = subprocess.run(
            scp_command(str(node["ssh_target"]), remote_tar, local_tar, ssh_options),
            text=True,
            capture_output=True,
            check=False,
        )
        result["scp_returncode"] = scp.returncode
        result["scp_stdout"] = scp.stdout
        result["scp_stderr"] = scp.stderr
        if scp.returncode == 0:
            result["collected_artifact"] = str(local_tar)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render or run guarded LMS fleet rollout scripts")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("render", "run"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--config", required=True)
        cmd.add_argument("--node", action="append", default=[])
        cmd.add_argument("--all-nodes", action="store_true")
        cmd.add_argument("--run-id", default=None)
        cmd.add_argument("--execute-candidate", action="append", default=[])
        cmd.add_argument("--update-code", action="store_true")
        cmd.add_argument("--dry-run-limit", type=int, default=4)
        cmd.add_argument("--output-dir", required=True)
        cmd.add_argument("--ssh-option", action="append", default=["BatchMode=yes", "ConnectTimeout=10"])
        cmd.add_argument("--continue-on-error", action="store_true")
        if name == "run":
            cmd.add_argument("--no-collect", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_json(args.config)
    nodes = resolve_nodes(config, args.node, args.all_nodes)
    execute_map = parse_execute_candidates(args.execute_candidate)
    selected_ids = {str(node["node_id"]) for node in nodes}
    unknown_execute_nodes = sorted(set(execute_map) - selected_ids)
    if unknown_execute_nodes:
        raise SystemExit(
            "execute candidates reference unselected nodes: " + ", ".join(unknown_execute_nodes)
        )
    run_id = args.run_id or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir = output_dir / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)

    scripts: Dict[str, str] = {}
    for node in nodes:
        node_id = str(node["node_id"])
        script = build_remote_script(
            node,
            run_id,
            execute_candidates=execute_map.get(node_id, []),
            update_code=args.update_code,
            dry_run_limit=args.dry_run_limit,
        )
        scripts[node_id] = script
        path = scripts_dir / f"{safe_slug(node_id)}.sh"
        path.write_text(script, encoding="utf-8")
        path.chmod(0o700)

    manifest_core = {
        "run_id": run_id,
        "node_ids": [str(node["node_id"]) for node in nodes],
        "execute_candidates": execute_map,
        "update_code": bool(args.update_code),
        "mode": args.command,
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "fleet_rollout",
        "created_at_utc": utc_now_iso(),
        **manifest_core,
        "rollout_fingerprint": canonical_hash(manifest_core),
    }
    write_json(str(output_dir / "rollout_manifest.json"), manifest)

    if args.command == "render":
        print(f"wrote {len(scripts)} guarded rollout script(s) under {scripts_dir}")
        return 0

    results: List[Dict[str, Any]] = []
    for node in nodes:
        result = execute_remote(
            node,
            scripts[str(node["node_id"])],
            run_id,
            output_dir / "artifacts",
            args.ssh_option,
            collect=not args.no_collect,
        )
        results.append(result)
        if result["returncode"] != 0 and not args.continue_on_error:
            break
    write_json(
        str(output_dir / "rollout_results.json"),
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "fleet_rollout_results",
            "run_id": run_id,
            "created_at_utc": utc_now_iso(),
            "results": results,
        },
    )
    failures = [item for item in results if item.get("returncode") != 0]
    print(f"wrote {output_dir / 'rollout_results.json'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
