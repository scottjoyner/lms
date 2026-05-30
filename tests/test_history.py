import csv
import json

import lms_history


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_run(tmp_path, run_id="run-a", score="0.9"):
    run = tmp_path / run_id
    run.mkdir()
    (run / "lms_run_config.json").write_text(json.dumps({"run_id": run_id, "created_at_utc": "2026-01-01T00:00:00Z", "endpoints": ["http://127.0.0.1:1234/v1"]}))
    (run / "run_audit.json").write_text(json.dumps({"status": "pass", "ok": True}))
    (run / "machine_profile.json").write_text(json.dumps({"host": {"hostname": "host1", "platform": "linux"}}))
    write_csv(
        run / "capability_matrix.csv",
        [{
            "run_id": run_id,
            "task_family": "coding",
            "model_key": "qwen-7b-q4",
            "base_url": "http://127.0.0.1:1234/v1",
            "host_name": "host1",
            "score": score,
            "grade": "A",
            "reliability_grade": "A",
            "throughput_grade": "B",
            "latency_grade": "A",
            "max_reliable_context_tokens": "8192",
            "evidence": "ok",
            "recommended_use": "preferred",
            "avoid_use": "",
        }],
        ["run_id", "task_family", "model_key", "base_url", "host_name", "score", "grade", "reliability_grade", "throughput_grade", "latency_grade", "max_reliable_context_tokens", "evidence", "recommended_use", "avoid_use"],
    )
    return run


def test_ingest_and_query_best(tmp_path):
    db = tmp_path / "history.sqlite3"
    run = make_run(tmp_path)
    with lms_history.connect(db) as conn:
        result = lms_history.ingest_run(conn, run)
        assert result["capability_rows"] == 1
        rows = lms_history.best_routes(conn, task="coding", limit=10, min_score=0.0)
    assert len(rows) == 1
    assert rows[0]["model_key"] == "qwen-7b-q4"
    assert rows[0]["audit_status"] == "pass"


def test_ingest_many_from_parent(tmp_path):
    db = tmp_path / "history.sqlite3"
    make_run(tmp_path, "run-a", "0.7")
    make_run(tmp_path, "run-b", "0.9")
    with lms_history.connect(db) as conn:
        results = lms_history.ingest_many(conn, [tmp_path])
        assert len(results) == 2
        rows = lms_history.list_runs(conn, limit=10)
        assert len(rows) == 2
