"""Installed entrypoint for guarded physical fleet rollout.

The wrapper guarantees one failure-safe artifact packaging pass on remote exit
and attempts collection even when the remote benchmark fails. Benchmark and
packaging failures remain visible to operators and automation.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from lms_agent_bench import fleet_rollout as _base

_ORIGINAL_BUILD_REMOTE_SCRIPT = _base.build_remote_script
_ORIGINAL_EXECUTE_REMOTE = _base.execute_remote


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
    script = _remove_legacy_packaging(script)
    marker = 'mkdir -p "$ARTIFACT_DIR"\n'
    if marker not in script:
        raise RuntimeError(
            "rollout script no longer exposes the artifact-directory marker"
        )
    return script.replace(
        marker, marker + _exit_packaging_snippet(), 1
    )


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
    local_tar = collect_dir / (
        f"{_base.safe_slug(str(node['node_id']))}.tar.gz"
    )
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


def main(argv: Optional[List[str]] = None) -> int:
    _base.build_remote_script = build_remote_script
    _base.execute_remote = execute_remote
    return _base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
