from __future__ import annotations

import sys
from pathlib import Path

from lms_agent_bench import benchmark_entrypoint, lms_eval
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


def test_benchmark_entrypoint_uses_real_package_evaluator():
    import lms_agent_bench.benchmark_lmstudio_cross_machine_models as runner

    assert sys.modules["lms_eval"] is lms_eval
    assert runner.evaluate_output is lms_eval.evaluate_output
    assert benchmark_entrypoint.main is runner.main
