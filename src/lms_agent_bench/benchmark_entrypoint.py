"""Installed reliability-first benchmark entrypoint.

The historical runner is retained as an isolated single-trial measurement
engine. The public entrypoint adds strict identity checks, deterministic
multi-trial execution, whole-trial retries, robust statistics, verified raw
sample artifacts, and fingerprint-sealed resumability.
"""
from __future__ import annotations

from lms_agent_bench.benchmark_reliable_hardened import main

__all__ = ["main"]

if __name__ == "__main__":
    raise SystemExit(main())
