#!/usr/bin/env python3
"""Build a real LMS wheel offline and smoke installed CLIs outside the repo.

Source-only packaging gate. Never starts a model/server, checks fleet nodes,
modifies admission, reads secrets, or contacts the network. Dependencies are
resolved from the CI runner's existing local site-packages; the LMS wheel
itself is installed into an isolated, disposable virtual environment.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SMOKE_COMMANDS = (
    "lms-bench",
    "lms-bench-endpoints",
    "lms-fleet",
    "lms-fleet-gate",
    "lms-runtime-canary",
    "lms-runtime-canary-attest",
    "lms-loadout-qualification-run",
    "lms-agent",
)
CHECK_RESOURCES = (
    ("schemas", "fleet_loadout.v1.schema.json"),
    ("benchmarks", "agent_skill_suite.v1.json"),
)


def checked(cmd: list[str], *, cwd: Path, env: dict[str, str], timeout: int = 90) -> None:
    result = subprocess.run(
        cmd, cwd=cwd, env=env, capture_output=True, text=True,
        check=False, timeout=timeout,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout)[-1800:]
        raise RuntimeError(f"offline packaging smoke failed ({result.returncode}): {cmd[0]}: {detail}")


def run() -> dict[str, object]:
    env = dict(os.environ)
    env.update(
        PIP_NO_INDEX="1",
        PIP_DISABLE_PIP_VERSION_CHECK="1",
        PYTHONNOUSERSITE="1",
        PYTHONPATH="",
    )
    with tempfile.TemporaryDirectory(prefix="lms-wheel-test-") as folder:
        outside = Path(folder)
        wheelhouse = outside / "wheels"
        wheelhouse.mkdir()
        checked(
            [sys.executable, "-m", "pip", "wheel", "--no-index",
             "--no-build-isolation", "--no-deps", "--wheel-dir",
             str(wheelhouse), str(ROOT)],
            cwd=outside, env=env,
        )
        wheels = sorted(wheelhouse.glob("lms_agent_bench-*.whl"))
        if len(wheels) != 1:
            raise RuntimeError("exactly one built wheel required")
        venv = outside / "venv"
        # CI installs declared dependencies already; use its local wheel/dep
        # packages, but install the LMS wheel ahead of the editable CI copy.
        checked([sys.executable, "-m", "venv", "--system-site-packages",
                 str(venv)], cwd=outside, env=env)
        py = venv / "bin" / "python"
        checked(
            [str(py), "-m", "pip", "install", "--no-index", "--no-deps",
             "--ignore-installed", str(wheels[0])],
            cwd=outside, env=env,
        )
        payload = (
            "import json, pathlib, sys; "
            "from importlib.metadata import distribution; "
            "from importlib.resources import files; "
            "import lms_agent_bench; "
            "root=pathlib.Path(lms_agent_bench.__file__).resolve(); "
            "assert str(root).startswith(str(pathlib.Path(sys.prefix).resolve())), root; "
            "d=distribution('lms-agent-bench'); "
            "r=files('lms_agent_bench'); "
            "names=" + repr(CHECK_RESOURCES) + "; "
            "assert all(r.joinpath(folder,name).is_file() for folder,name in names); "
            "print(json.dumps({'wheel_version':d.version,'package_path':str(root),"
            "'resource_count':len(names)}))"
        )
        result = subprocess.run(
            [str(py), "-I", "-c", payload],
            cwd=outside, env=env, capture_output=True, text=True,
            timeout=12, check=True,
        )
        manifest = json.loads(result.stdout.strip())
        for name in SMOKE_COMMANDS:
            checked([str(venv / "bin" / name), "--help"],
                    cwd=outside, env=env, timeout=15)
        # Only safe summary. No temporary installation paths or private data.
        return {
            "status": "PASS",
            "network_used": False,
            "model_forward_passes": 0,
            "production_deployment": False,
            "version": manifest["wheel_version"],
            "cli_entrypoints_checked": list(SMOKE_COMMANDS),
            "packaged_resources_checked": manifest["resource_count"],
        }


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
