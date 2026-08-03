from lms_agent_bench.fleet_models import (
    fingerprint_inventory,
    parse_parameter_billions,
    parse_quantization,
    scan_model_roots,
    selection_model_id,
)

import pytest


def test_scan_gguf_and_upgrade_selected_hash(tmp_path):
    model = tmp_path / "Qwen3.5-9B-Q4_K_M.gguf"
    model.write_bytes(b"model-bytes")
    inventory = scan_model_roots([str(tmp_path)], hash_mode="quick", default_max_context=16384)
    assert len(inventory["models"]) == 1
    record = inventory["models"][0]
    assert record["quantization"] == "Q4_K_M"
    assert record["parameter_billions"] == 9.0
    assert record["max_context"] == 16384
    assert record["quick_fingerprint"].startswith("sha256:")
    assert "content_sha256" not in record

    upgraded = fingerprint_inventory(inventory, [record["id"]])
    selected = upgraded["models"][0]
    assert selected["fingerprint_mode"] == "full"
    assert selected["content_sha256"].startswith("sha256:")
    assert selected["artifact_fingerprint"] == selected["content_sha256"]


def test_onnx_genai_directory_is_one_model(tmp_path):
    model = tmp_path / "Llama-3.2-3B-Instruct-INT4"
    model.mkdir()
    (model / "genai_config.json").write_text("{}")
    (model / "tokenizer.json").write_text("{}")
    (model / "model.onnx").write_bytes(b"onnx")
    inventory = scan_model_roots([str(tmp_path)], hash_mode="quick")
    assert [item["id"] for item in inventory["models"]] == [model.name]
    record = inventory["models"][0]
    assert record["format"] == "onnx-genai"
    assert record["quantization"] == "INT4"
    assert record["parameter_billions"] == 3.0


def test_selection_model_id_requires_selected_candidate():
    selection = {"selected": {"candidate": {"model": {"id": "model.gguf"}}}}
    assert selection_model_id(selection) == "model.gguf"
    with pytest.raises(ValueError):
        selection_model_id({"selected": None})


def test_filename_parsers_are_conservative():
    assert parse_quantization("model-IQ4_XS.gguf") == "IQ4_XS"
    assert parse_quantization("model.bin") == "unknown"
    assert parse_parameter_billions("model-0.5B-Q4_K_M.gguf") == 0.5
