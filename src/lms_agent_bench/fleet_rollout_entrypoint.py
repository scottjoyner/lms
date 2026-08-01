"""Installed entrypoint for guarded physical fleet rollout.

This wrapper is the production rollout surface. It expands explicit environment
placeholders, validates Tier-1 configuration, routes generated remote commands
through the hardened planner/executor entrypoints, and guarantees one
failure-safe artifact packaging pass on every remote exit.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

from lms_agent_bench import fleet_bench_entrypoint
from lms_agent_bench import fleet_rollout as _base

_ORIGINAL_BUILD_REMOTE_SCRIPT = _base.build_remote_script
_ORIGINAL_EXECUTE_REMOTE = _base.execute_remote
_ORIGINAL_LOAD_JSON = _base.load_json
_PLACEHOLDER_RE = re.compile(
    r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))"
)


def _exit_packaging_snippet() -> str:
    return r'''lms_fleet_package_artifacts() {
  status=$?
  package_status=0
  trap - EXIT
  set +e
  if [ -n "${ARTIFACT_DIR:-}" ] && [ -d "$ARTIFACT_DIR" ]; then
    rm -f "$ARTIFACT_DIR/bundle_manifest.json"
    "$PYTHON_BIN" - "$ARTIFACT_DIR" "$status" <<'PY' || package_status=$?
import hashlib, json, os, pathlib, sys
root = pathlib.Path(sys.argv[1])
remote_exit_code = int(sys.argv[2])
files = []
for path in sorted(
    p for p in root.rglob('*')
    if p.is_file() and p.name != 'bundle_manifest.json'
):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    files.append({
        'path': str(path.relative_to(root)),
        'size_bytes': path.stat().st_size,
        'sha256': 'sha256:' + digest.hexdigest(),
    })
manifest = {
    'schema_version': 'fleet_artifact_bundle.v1',
    'node_id': os.environ.get('NODE_ID'),
    'remote_exit_code': remote_exit_code,
    'files': files,
}
(root / 'bundle_manifest.json').write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + '\n',
    encoding='utf-8',
)
PY
    if [ "$package_status" -eq 0 ]; then
      tar -C "$ARTIFACT_DIR" -czf "$ARTIFACT_DIR.tar.gz" . || package_status=$?
    fi
    if [ "$package_status" -eq 0 ]; then
      echo "LMS_FLEET_ARTIFACT=$ARTIFACT_DIR.tar.gz"
    else
      echo "LMS fleet artifact packaging failed with code $package_status" >&2
    fi
  fi
  if [ "$status" -eq 0 ] && [ "$package_status" -ne 0 ]; then
    status=$package_status
  fi
  exit "$status"
}
trap lms_fleet_package_artifacts EXIT
'''


def _remove_legacy_packaging(script: str) -> str:
    marker = (
        '$PYTHON_BIN - "$ARTIFACT_DIR" <<\'PY\'\n'
        "import hashlib, json, os, pathlib, sys\n"
    )
    index = script.rfind(marker)
    if index < 0:
        raise RuntimeError(
            "rollout script no longer exposes the legacy packaging block"
        )
    return script[:index].rstrip() + "\n"


def _route_hardened_entrypoints(script: str) -> str:
    required_replacements = {
        "lms_agent_bench.fleet_loadout discover": (
            "lms_agent_bench.fleet_loadout_entrypoint discover"
        ),
        "lms_agent_bench.fleet_loadout plan": (
            "lms_agent_bench.fleet_loadout_entrypoint plan"
        ),
        "lms_agent_bench.fleet_bench_plan": (
            "lms_agent_bench.fleet_bench_entrypoint"
        ),
    }
    for original, hardened in required_replacements.items():
        if original not in script:
            raise RuntimeError(
                f"rollout script no longer contains expected command: {original}"
            )
        script = script.replace(original, hardened)

    optional_replacements = {
        "lms_agent_bench.fleet_loadout select": (
            "lms_agent_bench.fleet_loadout_entrypoint select"
        )
    }
    for original, hardened in optional_replacements.items():
        if original in script:
            script = script.replace(original, hardened)
    return script


def build_remote_script(
    node: Mapping[str, Any],
    run_id: str,
    execute_candidates: Sequence[str] = (),
    update_code: bool = False,
    dry_run_limit: int = 4,
) -> str:
    script = _ORIGINAL_BUILD_REMOTE_SCRIPT(
        node,
        run_id,
        execute_candidates=execute_candidates,
        update_code=update_code,
        dry_run_limit=dry_run_limit,
    )
    script = _route_hardened_entrypoints(script)
    script = _remove_legacy_packaging(script)
    marker = 'mkdir -p "$ARTIFACT_DIR"\n'
    if marker not in script:
        raise RuntimeError(
            "rollout script no longer exposes the artifact-directory marker"
        )
    return script.replace(marker, marker + _exit_packaging_snippet(), 1)


def execute_remote(
    node: Mapping[str, Any],
    script: str,
    run_id: str,
    collect_dir: Path,
    ssh_options: Sequence[str],
    collect: bool,
) -> Dict[str, Any]:
    result = _ORIGINAL_EXECUTE_REMOTE(
        node,
        script,
        run_id,
        collect_dir,
        ssh_options,
        collect=False,
    )
    if not collect:
        return result

    remote_tar = _base.remote_artifact_path(node, run_id) + ".tar.gz"
    collect_dir.mkdir(parents=True, exist_ok=True)
    local_tar = collect_dir / f"{_base.safe_slug(str(node['node_id']))}.tar.gz"
    scp = subprocess.run(
        _base.scp_command(
            str(node["ssh_target"]),
            remote_tar,
            local_tar,
            ssh_options,
        ),
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


def load_env_file(path: Optional[str]) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path:
        return values
    source = Path(path)
    for line_number, raw in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            raise ValueError(
                f"invalid environment assignment at {source}:{line_number}"
            )
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(
                f"invalid environment variable {key!r} at "
                f"{source}:{line_number}"
            )
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {'"', "'"}
        ):
            value = value[1:-1]
        values[key] = value
    return values


def _expand_string(
    value: str, variables: Mapping[str, str], location: str
) -> str:
    missing: List[str] = []

    def replace(match: re.Match[str]) -> str:
        key = match.group(1) or match.group(2)
        if key not in variables or variables[key] == "":
            missing.append(key)
            return match.group(0)
        return variables[key]

    expanded = _PLACEHOLDER_RE.sub(replace, value)
    if missing:
        raise ValueError(
            f"unresolved environment variable(s) at {location}: "
            + ", ".join(sorted(set(missing)))
        )
    return expanded


def expand_config_value(
    value: Any, variables: Mapping[str, str], location: str = "$"
) -> Any:
    if isinstance(value, str):
        return _expand_string(value, variables, location)
    if isinstance(value, list):
        return [
            expand_config_value(item, variables, f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        return {
            str(key): expand_config_value(
                item, variables, f"{location}.{key}"
            )
            for key, item in value.items()
        }
    return value


def load_rollout_config(path: str, env_file: Optional[str]) -> Dict[str, Any]:
    raw = _ORIGINAL_LOAD_JSON(path)
    variables: MutableMapping[str, str] = dict(os.environ)
    variables.update(load_env_file(env_file))
    expanded = expand_config_value(raw, variables)
    if not isinstance(expanded, dict):
        raise ValueError("rollout configuration must be a JSON object")
    return expanded


def validate_resolved_config(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    _base.validate_config(config)
    findings: List[Dict[str, Any]] = []
    for node in config["nodes"]:
        node_id = str(node["node_id"])
        errors: List[str] = []
        warnings: List[str] = []
        ssh_target = str(node.get("ssh_target") or "")
        repo_dir = str(node.get("repo_dir") or "")
        python_bin = str(node.get("python") or "python3")
        model_roots = [str(item) for item in node.get("model_roots", [])]
        endpoint_map = {
            str(key): str(value)
            for key, value in (node.get("endpoint_map") or {}).items()
        }
        if any(ch.isspace() for ch in ssh_target):
            errors.append("ssh_target contains whitespace")
        if not repo_dir.startswith("/"):
            errors.append("repo_dir must be an absolute remote path")
        if "/" in python_bin and not python_bin.startswith("/"):
            errors.append(
                "python must be a command name or an absolute remote path"
            )
        if len(model_roots) != len(set(model_roots)):
            errors.append("model_roots contains duplicates")
        for root in model_roots:
            if not root.startswith(("/", "~")):
                errors.append(
                    f"model root must be absolute or home-relative: {root}"
                )
        for candidate_id, url in endpoint_map.items():
            if not fleet_bench_entrypoint.is_loopback_url(url):
                errors.append(
                    f"endpoint_map {candidate_id} is not loopback-local: {url}"
                )
        if not node.get("metadata"):
            warnings.append("node has no descriptive metadata")
        findings.append(
            {
                "node_id": node_id,
                "ssh_target": ssh_target,
                "ready_for_observation": not errors,
                "errors": errors,
                "warnings": warnings,
                "metadata": node.get("metadata", {}),
            }
        )
    return findings


def validate_command(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="lms-fleet-rollout validate",
        description="Resolve and statically validate a rollout configuration",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--node", action="append", default=[])
    parser.add_argument("--all-nodes", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(list(argv))

    try:
        config = load_rollout_config(args.config, args.env_file)
        nodes = _base.resolve_nodes(config, args.node, args.all_nodes)
        selected_config = {
            "schema_version": config["schema_version"],
            "nodes": nodes,
        }
        findings = validate_resolved_config(selected_config)
        ready = all(item["ready_for_observation"] for item in findings)
        report = {
            "schema_version": "fleet_rollout_validation.v1",
            "created_at_utc": _base.utc_now_iso(),
            "config": str(Path(args.config)),
            "env_file": str(Path(args.env_file)) if args.env_file else None,
            "node_ids": [item["node_id"] for item in findings],
            "ready_for_observation": ready,
            "findings": findings,
            "admission": {"admitted": False},
        }
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        report = {
            "schema_version": "fleet_rollout_validation.v1",
            "created_at_utc": _base.utc_now_iso(),
            "config": str(Path(args.config)),
            "env_file": str(Path(args.env_file)) if args.env_file else None,
            "node_ids": [],
            "ready_for_observation": False,
            "findings": [],
            "error": str(exc),
            "admission": {"admitted": False},
        }
        ready = False

    _base.write_json(args.out, report)
    print(f"wrote {args.out}")
    return 0 if ready else 1


def _extract_env_file(argv: Sequence[str]) -> tuple[List[str], Optional[str]]:
    cleaned: List[str] = []
    env_file: Optional[str] = None
    index = 0
    values = list(argv)
    while index < len(values):
        value = values[index]
        if value == "--env-file":
            if index + 1 >= len(values):
                raise SystemExit("--env-file requires a path")
            env_file = values[index + 1]
            index += 2
            continue
        if value.startswith("--env-file="):
            env_file = value.split("=", 1)[1]
            index += 1
            continue
        cleaned.append(value)
        index += 1
    return cleaned, env_file


def main(argv: Optional[List[str]] = None) -> int:
    actual_argv = list(sys.argv[1:] if argv is None else argv)
    if actual_argv and actual_argv[0] == "validate":
        return validate_command(actual_argv[1:])

    cleaned_argv, env_file = _extract_env_file(actual_argv)
    _base.build_remote_script = build_remote_script
    _base.execute_remote = execute_remote
    _base.load_json = lambda path: load_rollout_config(path, env_file)
    return _base.main(cleaned_argv)


if __name__ == "__main__":
    raise SystemExit(main())
