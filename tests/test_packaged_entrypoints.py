from __future__ import annotations

import re
import sys
from pathlib import Path

import lms_agent_bench
from lms_agent_bench import (
    benchmark_entrypoint,
    benchmark_raw_entrypoint,
    benchmark_reliable_hardened,
    lms_eval,
)
from lms_agent_bench.fleet_bench_entrypoint import (
    default_suite_file,
    inject_default_suite,
)


def test_default_suite_is_packaged_and_injected():
    suite = default_suite_file()
    assert Path(suite).exists()
    args = ["--plan", "p", "--candidate", "c", "--output-dir", "o"]
    assert inject_default_suite(args)[-2:] == ["--suite-file", suite]
    explicit = ["--suite-file", "custom.json"]
    assert inject_default_suite(explicit) == explicit


def test_benchmark_entrypoints_use_real_evaluator_and_hardened_orchestration():
    import lms_agent_bench.benchmark_lmstudio_cross_machine_models as runner

    assert sys.modules["lms_eval"] is lms_eval
    assert runner.evaluate_output is lms_eval.evaluate_output
    assert benchmark_raw_entrypoint.main is runner.main
    assert benchmark_entrypoint.main is benchmark_reliable_hardened.main


def test_importable_version_matches_project_metadata():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'(?m)^version = "([^"]+)"$', pyproject)
    assert match is not None
    assert lms_agent_bench.__version__ == match.group(1)
