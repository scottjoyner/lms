import csv
import json

from lms_decide import decide


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_run(tmp_path, *, score="0.9", fit_grade="good", max_ctx="8192", audit_status="pass"):
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_audit.json").write_text(json.dumps({"status": audit_status, "ok": audit_status == "pass", "critical": ["bad"] if audit_status == "fail" else [], "warnings": ["warn"] if audit_status == "warn" else []}))
    write_csv(
        run / "capability_matrix.csv",
        [
            {
                "task_family": "coding",
                "model_key": "qwen-7b-q4",
                "base_url": "http://127.0.0.1:1234/v1",
                "host_name": "host",
                "score": score,
                "grade": "A",
                "max_reliable_context_tokens": max_ctx,
                "evidence": "ok",
            },
            {
                "task_family": "coding",
                "model_key": "fallback-7b-q4",
                "base_url": "http://127.0.0.1:1234/v1",
                "host_name": "host",
                "score": "0.7",
                "grade": "B",
                "max_reliable_context_tokens": "4096",
                "evidence": "fallback",
            },
        ],
        ["task_family", "model_key", "base_url", "host_name", "score", "grade", "max_reliable_context_tokens", "evidence"],
    )
    write_csv(
        run / "model_fit.csv",
        [
            {"model_key": "qwen-7b-q4", "fit_grade": fit_grade, "fit_notes": "fit note"},
            {"model_key": "fallback-7b-q4", "fit_grade": "good", "fit_notes": "good"},
        ],
        ["model_key", "fit_grade", "fit_notes"],
    )
    return run


def test_decide_allows_good_route(tmp_path):
    run = make_run(tmp_path)
    result = decide(run, task="coding", context_tokens=2048, min_score=0.55)
    assert result["decision"] == "allow"
    assert result["selected"]["model_key"] == "qwen-7b-q4"


def test_decide_uses_fallback_when_context_too_large(tmp_path):
    run = make_run(tmp_path, max_ctx="1024")
    result = decide(run, task="coding", context_tokens=2048, min_score=0.55)
    assert result["decision"] == "allow"
    assert result["selected"]["model_key"] == "fallback-7b-q4"


def test_decide_blocks_failed_audit(tmp_path):
    run = make_run(tmp_path, audit_status="fail")
    result = decide(run, task="coding", context_tokens=2048, min_score=0.55)
    assert result["decision"] == "block"
    assert "audit" in result["reason"]


def test_decide_blocks_poor_fit_when_no_good_route(tmp_path):
    run = make_run(tmp_path, fit_grade="poor")
    # Make fallback too low score so the poor-fit preferred route cannot be replaced.
    rows = [r for r in csv.DictReader((run / "capability_matrix.csv").open())]
    rows[1]["score"] = "0.1"
    write_csv(run / "capability_matrix.csv", rows, rows[0].keys())
    result = decide(run, task="coding", context_tokens=2048, min_score=0.55)
    assert result["decision"] == "block"
