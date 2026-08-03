from __future__ import annotations

import copy
import json
from pathlib import Path

from lms_agent_bench import hermes_agent_bench
from lms_agent_bench.hermes_agent_common import canonical_hash, validate_suite
from lms_agent_bench.hermes_agent_scoring import (
    aggregate_trials,
    evaluate_gate,
    evaluate_trial,
)
from lms_agent_bench.model_loadout import validate_manifest


def loadout():
    raw = json.loads(
        Path("examples/model-loadouts.v1.example.json").read_text(encoding="utf-8")
    )["base_manifests"][0]
    return validate_manifest(raw)


def suite():
    raw = json.loads(
        Path("src/lms_agent_bench/benchmarks/hermes_agent_suite.v1.json").read_text(
            encoding="utf-8"
        )
    )
    return validate_suite(raw)


def test_context_pressure_scales_with_exact_configured_context():
    case = {
        "context_pressure_ratio": 0.5,
        "context_control_code": "CONTROL-123",
    }
    small = loadout()
    large = copy.deepcopy(small)
    large.pop("loadout_fingerprint")
    large.pop("derived")
    large.pop("admission")
    large["context"]["configured_tokens"] = 16_384
    large["kv_cache"]["capacity_tokens"] = 16_384
    large = validate_manifest(large)

    small_text = hermes_agent_bench.context_pressure_text(case, small)
    large_text = hermes_agent_bench.context_pressure_text(case, large)

    assert "CONTROL-123" in small_text
    assert len(large_text) >= len(small_text) * 1.9


def test_effect_scoring_requires_file_result_and_successful_test(tmp_path):
    case = next(item for item in suite()["cases"] if item["case_key"] == "mcp_code_edit_test_loop")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "calculator.py").write_text(
        "def add(a, b):\n    return a + b\n", encoding="utf-8"
    )
    calls = [
        {"tool": "workspace_read", "argument_valid": True, "is_error": False},
        {"tool": "workspace_patch", "argument_valid": True, "is_error": False},
        {"tool": "workspace_run_tests", "argument_valid": True, "is_error": False},
    ]
    trial = evaluate_trial(
        case=case,
        trial_index=1,
        process_result={
            "ok": True,
            "wall_seconds": 4.0,
            "result": {
                "final_response": "The tests passed.",
                "messages": [],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                },
            },
        },
        process_returncode=0,
        timed_out=False,
        stdout="",
        stderr="",
        fixture_calls=calls,
        workspace=workspace,
        prohibited_tools={"send_email", "graph_delete_all"},
    )

    assert trial["passed"] is True
    assert trial["checkpoint_rate"] == 1.0
    assert trial["completion_tokens_per_second"] == 5.0
    assert trial["tool_calls_per_minute"] == 45.0


def test_p0_and_recovery_cases_are_hard_gates():
    minimal_suite = {
        "minimum_valid_trials": 3,
        "gate": {
            "minimum_overall_task_pass_rate": 0.8,
            "minimum_effect_checkpoint_rate": 0.9,
            "minimum_argument_validity_rate": 0.95,
            "maximum_prohibited_tool_calls": 0,
            "require_all_p0_cases": True,
            "require_all_recovery_cases": True,
            "maximum_timeout_or_crash_rate": 0.0,
        },
        "cases": [
            {
                "case_key": "p0",
                "priority": "P0",
                "task_family": "tool",
                "recovery_case": False,
            },
            {
                "case_key": "recovery",
                "priority": "P1",
                "task_family": "recovery",
                "recovery_case": True,
            },
        ],
    }
    trials = []
    for case_key in ("p0", "recovery"):
        for trial_index in range(1, 4):
            trials.append(
                {
                    "case_key": case_key,
                    "valid": True,
                    "passed": True,
                    "wall_seconds": 1.0,
                    "tool_call_count": 1,
                    "invalid_argument_call_count": 0,
                    "tool_error_call_count": 0,
                    "prohibited_tool_calls": [],
                    "checkpoint_weight": 1.0,
                    "earned_checkpoint_weight": 1.0,
                    "usage": {"completion_tokens": 10},
                    "timed_out": False,
                }
            )
    aggregate = aggregate_trials(minimal_suite, trials)
    assert evaluate_gate(minimal_suite, aggregate)["passed"] is True

    trials[-1]["passed"] = False
    aggregate = aggregate_trials(minimal_suite, trials)
    gate = evaluate_gate(minimal_suite, aggregate)
    assert gate["passed"] is False
    assert any("recovery" in reason for reason in gate["failures"])


def test_dry_run_records_exact_loadout_but_cannot_qualify(tmp_path):
    manifest_path = tmp_path / "loadout.json"
    manifest_path.write_text(json.dumps(loadout()), encoding="utf-8")
    report_path = tmp_path / "report.json"

    rc = hermes_agent_bench.main(
        [
            "run",
            "--loadout",
            str(manifest_path),
            "--hermes-repo",
            str(tmp_path / "unused-hermes"),
            "--endpoint",
            "http://127.0.0.1:1234/v1",
            "--workspace",
            str(tmp_path / "runs"),
            "--dry-run",
            "--out",
            str(report_path),
        ]
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert rc == 1
    assert report["dry_run"] is True
    assert report["gate"]["passed"] is False
    assert report["identity"]["loadout_fingerprint"] == loadout()["loadout_fingerprint"]
    assert report["identity"]["architecture_kind"] == "dense"
    assert report["admission"]["admitted"] is False


def test_report_fingerprint_binds_loadout_identity():
    exact = loadout()
    identity = {
        "node_id": exact["node_id"],
        "candidate_id": exact["candidate_id"],
        "model_id": exact["model"]["id"],
        "model_content_sha256": exact["model"]["content_sha256"],
        "loadout_fingerprint": exact["loadout_fingerprint"],
        "loopback_only": True,
    }
    core = {
        "identity": identity,
        "suite_id": "test",
        "suite_fingerprint": "sha256:" + "1" * 64,
        "trials_per_case": 3,
        "trials": [],
        "aggregate": {},
        "gate": {"passed": True},
        "dry_run": False,
        "admission": {"admitted": False},
    }
    first = canonical_hash(core)
    identity["loadout_fingerprint"] = "sha256:" + "2" * 64
    assert canonical_hash(core) != first
