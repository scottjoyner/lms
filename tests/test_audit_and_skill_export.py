import csv
import json

from lms_run_audit import audit_run
from lms_skill_export import export_skill


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_minimal_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "lms_run_config.json").write_text(json.dumps({"run_id": "test", "created_at_utc": "now", "endpoints": ["http://127.0.0.1:1234/v1"]}))
    (run / "machine_profile.json").write_text(json.dumps({"host": {"hostname": "test", "platform": "linux", "python_version": "3.12"}}))
    write_csv(run / "lmstudio_inventory.csv", [{"host_name": "test", "host_ip": "127.0.0.1", "endpoint_id": 1, "base_url": "http://127.0.0.1:1234/v1", "reachable": 1, "model_id": 1, "model_key": "qwen-7b-q4"}], ["host_name", "host_ip", "endpoint_id", "base_url", "reachable", "model_id", "model_key"])
    write_csv(run / "run_results.csv", [{"phase": "run", "ok": "True", "eval_ok": "True", "task_family": "safety", "context_tokens": "", "model_key": "qwen-7b-q4", "base_url": "http://127.0.0.1:1234/v1"}], ["phase", "ok", "eval_ok", "task_family", "context_tokens", "model_key", "base_url"])
    write_csv(run / "run_summary.csv", [{"run_id": "test", "host_name": "test", "host_ip": "127.0.0.1", "base_url": "http://127.0.0.1:1234/v1", "model_key": "qwen-7b-q4", "ok_rate": "1.00", "eval_ok_rate": "1.00", "eval_score_avg": "1.00", "ttft_med": "1", "tps_med": "20"}], ["run_id", "host_name", "host_ip", "base_url", "model_key", "ok_rate", "eval_ok_rate", "eval_score_avg", "ttft_med", "tps_med"])
    write_csv(run / "task_summary.csv", [{"run_id": "test", "host_name": "test", "host_ip": "127.0.0.1", "base_url": "http://127.0.0.1:1234/v1", "model_key": "qwen-7b-q4", "task_family": "safety", "ok_rate": "1.00", "eval_ok_rate": "1.00", "eval_score_avg": "1.00", "ttft_med": "1", "tps_med": "20"}], ["run_id", "host_name", "host_ip", "base_url", "model_key", "task_family", "ok_rate", "eval_ok_rate", "eval_score_avg", "ttft_med", "tps_med"])
    write_csv(run / "capability_matrix.csv", [{"run_id": "test", "host_name": "test", "host_ip": "127.0.0.1", "base_url": "http://127.0.0.1:1234/v1", "model_key": "qwen-7b-q4", "task_family": "safety", "score": "0.9", "grade": "A", "max_reliable_context_tokens": "", "evidence": "ok"}], ["run_id", "host_name", "host_ip", "base_url", "model_key", "task_family", "score", "grade", "max_reliable_context_tokens", "evidence"])
    (run / "routing_rules.json").write_text(json.dumps({"routing": {"safety": {"preferred": {"model_key": "qwen-7b-q4", "score": "0.9"}, "fallback": None, "task_family": "safety"}}}))
    (run / "routing_rules.yaml").write_text("routing: {}\n")
    write_csv(run / "model_fit.csv", [{"model_key": "qwen-7b-q4", "fit_grade": "good", "estimated_model_memory_gib": "4", "fit_notes": "ok"}], ["model_key", "fit_grade", "estimated_model_memory_gib", "fit_notes"])
    (run / "model_fit.md").write_text("ok")
    (run / "agent_recommendations.md").write_text("ok")
    (run / "AGENT_BRIEF.md").write_text("ok")
    (run / "sidecars").mkdir()
    return run


def test_audit_passes_minimal_run(tmp_path):
    run = make_minimal_run(tmp_path)
    audit = audit_run(run, min_score=0.55, min_eval_ok=0.6, require_safety=True)
    assert audit["status"] == "pass"


def test_skill_export_contains_routes(tmp_path):
    run = make_minimal_run(tmp_path)
    (run / "run_audit.json").write_text(json.dumps({"status": "pass", "ok": True, "critical": [], "warnings": []}))
    skill = export_skill(run)
    assert skill["schema_version"] == "lms_agent_skill.v1"
    assert "routes" in skill
    assert skill["audit"]["status"] == "pass"
