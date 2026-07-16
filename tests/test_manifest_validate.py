from pathlib import Path

from lms_manifest_validate import load_manifest, validate_manifest


def test_default_manifest_is_valid():
    result = validate_manifest(load_manifest(Path("benchmarks/agent_skill_suite.v1.json")))
    assert result["ok"] is True
    assert "safety" in result["task_families"]
    assert result["case_count"] >= 1


def test_duplicate_case_key_fails():
    data = {
        "cases": [
            {
                "case_key": "x",
                "priority": "P0",
                "task_family": "general",
                "system": "s",
                "prompt": "p",
                "max_output_tokens": 1,
                "evaluators": [{"type": "exact_contains", "value": "x"}],
                "recommendation_signal": "x",
            },
            {
                "case_key": "x",
                "priority": "P0",
                "task_family": "general",
                "system": "s",
                "prompt": "p",
                "max_output_tokens": 1,
                "evaluators": [{"type": "exact_contains", "value": "x"}],
                "recommendation_signal": "x",
            },
        ]
    }
    result = validate_manifest(data)
    assert result["ok"] is False
    assert any("duplicate" in err for err in result["errors"])
