"""Installed-package entrypoint for the canonical benchmark runner.

The historical runner imports ``lms_eval`` as a top-level module. An installed
src-layout package does not expose that name, so register the package evaluator
explicitly before importing the runner. This prevents the runner's permissive
fallback evaluator from silently marking every output as valid.
"""
from __future__ import annotations

import sys

from lms_agent_bench import lms_eval as _lms_eval

sys.modules["lms_eval"] = _lms_eval

from lms_agent_bench.benchmark_lmstudio_cross_machine_models import main

__all__ = ["main"]

if __name__ == "__main__":
    raise SystemExit(main())
