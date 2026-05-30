import csv
import json

import lms_history
from lms_history_decide import decide_from_history


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_run(tmp_path, run_id, *, score="0.9", max_ctx="8192", audit_status="pass"):
    run = tmp_path / run_id
    run.mkdir()
    (run / "lms_run_config.json").write_text(json.dumps({"run_id": run_id, "created_at_utc": "2026-01-01T00:00:00Z", "endpoints": ["http://127.0.0.1:1234/v1"]}))
    (run / "run_audit.json").write_text(json.dumps({"status": audit_status, "ok": audit_status == "pass"}))
    (run / "machine_profile.json").write_text(json.dumps({"host": {"hostname": "host1", "platform": "linux"}}))
    write_csv(
        run / "capability_matrix.csv",
        [{
            "run_id": run_id,
            "task_family": "coding",
            "model_key": f"model-{run_id}",
            "base_url": "http://127.0.0.1:1234/v1",
            "host_name": "host1",
            "score": score,
            "grade": "A",
            "reliability_grade": "A",
            "throughput_grade": "B",
            "latency_grade": "A",
            "max_reliable_context_tokens": max_ctx,
            "evidence": "ok",
            "recommended_use": "preferred",
            "avoid_use": "",
        }],
        ["run_id", "task_family", "model_key", "base_url", "host_name", "score", "grade", "reliability_grade", "throughput_grade", "latency_grade", "max_reliable_context_tokens", "evidence", "recommended_use", "avoid_use"],
    )
    return run


def test_history_decide_allows_best_historical_route(tmp_path):
    db = tmp_path / "history.sqlite3"
    run = make_run(tmp_path, "a", score="0.9", max_ctx="8192")
    with lms_history.connect(db) as conn:
        lms_history.ingest_run(conn, run)
    result = decide_from_history(db, task="coding", context_tokens=2048, min_score=0.55)
    assert result["decision"] == "allow"
    assert result["selected"]["model_key"] == "model-a"


def test_history_decide_blocks_when_context_exceeds_all_routes(tmp_path):
    db = tmp_path / "history.sqlite3"
    run = make_run(tmp_path, "a", score="0.9", max_ctx="1024")
    with lms_history.connect(db) as conn:
        lms_history.ingest_run(conn, run)
    result = decide_from_history(db, task="coding", context_tokens=4096, min_score=0.55)
    assert result["decision"] == "block"
    assert result["evaluated"]


def test_history_decide_skips_failed_audit_route(tmp_path):
    db = tmp_path / "history.sqlite3"
    failed = make_run(tmp_path, "failed", score="0.99", audit_status="fail")
    good = make_run(tmp_path, "good", score="0.8", audit_status="pass")
    with lms_history.connect(db) as conn:
        lms_history.ingest_run(conn, failed)
        lms_history.ingest_run(conn, good)
    result = decide_from_history(db, task="coding", context_tokens=1024, min_score=0.55)
    assert result["decision"] == "allow"
    assert result["selected"]["model_key"] == "model-good"
