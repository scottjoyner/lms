import json

from lms_artifact_validate import validate_one


def test_validate_one_passes_for_valid_audit(tmp_path):
    schema = tmp_path / "schema.json"
    artifact = tmp_path / "audit.json"
    schema.write_text(json.dumps({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["status", "ok"],
        "properties": {"status": {"type": "string"}, "ok": {"type": "boolean"}}
    }))
    artifact.write_text(json.dumps({"status": "pass", "ok": True}))
    result = validate_one(schema, artifact)
    assert result["ok"] is True


def test_validate_one_fails_for_missing_required(tmp_path):
    schema = tmp_path / "schema.json"
    artifact = tmp_path / "audit.json"
    schema.write_text(json.dumps({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["status", "ok"],
        "properties": {"status": {"type": "string"}, "ok": {"type": "boolean"}}
    }))
    artifact.write_text(json.dumps({"status": "pass"}))
    result = validate_one(schema, artifact)
    assert result["ok"] is False
    assert result["errors"]
