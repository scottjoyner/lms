"""Negative and positive preflight tests; never run a model or graph."""
import json
import pytest

from lms_agent_bench.hermes_wire_pair_preflight import (
    BRIDGE, DIRECT, PairRejected, compare,
)


def metadata(tools):
    return {
        "message_roles": ["system", "user"],
        "messages": [
            {"content_sha256": "1" * 64, "content_chars": 7000},
            {"content_sha256": "2" * 64, "content_chars": 157},
        ],
        "advertised_tools": [
            {"name": name, "description_sha256": name} for name in sorted(tools)
        ],
        "model_id": "liquid-24b-parity-candidate",
        "generation": {"temperature": 0.0, "max_tokens": 640, "stream": True},
        "raw_bodies_saved": False, "raw_headers_saved": False,
    }


def pair(tmp_path, override=None):
    a = metadata(BRIDGE)
    b = metadata(DIRECT)
    if override:
        override(a, b)
    bridge = tmp_path / "bridge.jsonl"
    direct = tmp_path / "direct.jsonl"
    bridge.write_text(json.dumps(a) + "\n")
    direct.write_text(json.dumps(b) + "\n")
    return bridge, direct


def test_matching_preambles_and_settings_only_admit_wire_comparison(tmp_path):
    a, b = pair(tmp_path)
    receipt = compare(a, b)
    assert receipt["wire_pair_admissible"] is True
    assert receipt["native_owner_proven"] is False
    assert receipt["promotion_authorized"] is False
    assert len(receipt["bridge"]["tool_names"]) == 3
    assert len(receipt["direct"]["tool_names"]) == 9


def test_rejects_preamble_drift_between_arms(tmp_path):
    a, b = pair(tmp_path)
    record = json.loads(b.read_text())
    record["messages"][0]["content_sha256"] = "4" * 64
    b.write_text(json.dumps(record) + "\n")
    with pytest.raises(PairRejected, match="system_sha256"):
        compare(a, b)


def test_rejects_decoding_drift_between_arms(tmp_path):
    a, b = pair(tmp_path)
    record = json.loads(b.read_text())
    record["generation"]["max_tokens"] = 320
    b.write_text(json.dumps(record) + "\n")
    with pytest.raises(PairRejected, match="bounded decoding"):
        compare(a, b)


def test_rejects_unsafe_capture(tmp_path):
    a, b = pair(tmp_path)
    record = json.loads(b.read_text())
    record["raw_bodies_saved"] = True
    b.write_text(json.dumps(record) + "\n")
    with pytest.raises(PairRejected, match="unsanitized"):
        compare(a, b)


def test_rejects_intra_arm_drift(tmp_path):
    a, b = pair(tmp_path)
    record = metadata(BRIDGE)
    record["messages"][0]["content_sha256"] = "a" * 64
    with a.open("a") as out:
        out.write(json.dumps(record) + "\n")
    with pytest.raises(PairRejected, match="preamble changed"):
        compare(a, b)


def test_rejects_wrong_direct_tool_count(tmp_path):
    a, b = pair(tmp_path)
    record = json.loads(b.read_text())
    record["advertised_tools"].pop()
    b.write_text(json.dumps(record) + "\n")
    with pytest.raises(PairRejected, match="surface"):
        compare(a, b)


def test_rejects_invalid_json(tmp_path):
    a, b = pair(tmp_path)
    b.write_text("{bad}\n")
    with pytest.raises(PairRejected, match="unparseable"):
        compare(a, b)
