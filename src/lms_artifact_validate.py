"""Backward-compatible import shim for lms_agent_bench.lms_artifact_validate."""
from lms_agent_bench.lms_artifact_validate import *  # noqa: F401,F403
from lms_agent_bench.lms_artifact_validate import main as _main

if __name__ == "__main__":
    raise SystemExit(_main())
