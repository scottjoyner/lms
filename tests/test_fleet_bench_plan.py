from __future__ import annotations

from pathlib import Path

import pytest

from lms_agent_bench.fleet_bench_plan import (
    build_llama_server_command,
    parse_endpoint_map,
    resolve_candidates,
)


def sample_plan():
    return {
        "candidates": [
            {"candidate_id": "a", "engine": "llama.cpp"},
            {"candidate_id": "b", "engine": "npu-inference-server"},
        ]
    }


def test_resolve_requires_explicit_scope():
    with pytest.raises(ValueError):
        resolve_candidates(sample_plan(), [], False, 0)
    assert [item["candidate_id"] for item in resolve_candidates(sample_plan(), ["b"], False, 0)] == ["b"]
    assert [item["candidate_id"] for item in resolve_candidates(sample_plan(), [], True, 1)] == ["a"]


def test_endpoint_map_normalizes_v1():
    assert parse_endpoint_map(["b=http://127.0.0.1:1236"]) == {
        "b": "http://127.0.0.1:1236/v1"
    }
    with pytest.raises(ValueError):
        parse_endpoint_map(["bad"])


def test_build_command_uses_candidate_loadout(tmp_path: Path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"x")
    candidate = {
        "candidate_id": "a",
        "model": {"path": str(model)},
        "benchmark_port": 18080,
        "context_tokens": 8192,
        "parallel_slots": 2,
        "threads": 8,
        "gpu_layers": 999,
        "batch_size": 512,
        "ubatch_size": 128,
        "flash_attention": True,
    }
    help_text = " ".join(
        [
            "--model",
            "--host",
            "--port",
            "--ctx-size",
            "--parallel",
            "--threads",
            "--n-gpu-layers",
            "--batch-size",
            "--ubatch-size",
            "--flash-attn",
        ]
    )
    command = build_llama_server_command(
        candidate, "/bin/llama-server", help_text
    )
    joined = " ".join(command)
    assert "--model" in command and str(model) in command
    assert "--ctx-size 8192" in joined
    assert "--parallel 2" in joined
    assert "--n-gpu-layers 999" in joined
    assert "--flash-attn on" in joined
