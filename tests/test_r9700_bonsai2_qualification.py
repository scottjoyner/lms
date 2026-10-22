import hashlib
from pathlib import Path

import pytest

from lms_agent_bench.r9700_bonsai2_qualification import (
    MODEL_ID,
    MODEL_PARAMETERS,
    RUNTIME_BUILD_MARKER,
    build_loadout,
    parse_runtime_arguments,
    validate_props,
)


def runtime_args(model_path: Path):
    return [
        "/opt/prism/llama-server",
        "--model",
        str(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--gpu-layers",
        "999",
        "--flash-attn",
        "on",
        "--ctx-size",
        "8192",
        "--parallel",
        "4",
        "--batch-size",
        "512",
        "--ubatch-size",
        "128",
    ]


def props(model_path: Path):
    return {
        "default_generation_settings": {
            "params": {"speculative.types": "none"},
            "n_ctx": 8192,
        },
        "total_slots": 4,
        "model_alias": str(model_path),
        "model_ftype": "PQ2_0 - 2.13 bpw (group 128)",
        "modalities": {"vision": False},
        "build_info": RUNTIME_BUILD_MARKER,
    }


def test_exact_runtime_and_props_build_valid_loadout(tmp_path):
    model = tmp_path / "Ternary-Bonsai-2-27B-PQ2_0.gguf"
    model.write_bytes(b"fixture")
    runtime = tmp_path / "llama-server"
    runtime.write_bytes(b"runtime")
    parsed = parse_runtime_arguments(
        runtime_args(model),
        model_path=model,
        endpoint="http://127.0.0.1:8000/v1",
    )
    observed = validate_props(
        props(model),
        model_path=model,
        runtime_arguments=parsed,
    )
    loadout = build_loadout(
        model_path=model,
        runtime_path=runtime,
        runtime_arguments=parsed,
        props_observation=observed,
    )
    assert loadout["model"]["id"] == MODEL_ID
    assert loadout["architecture"]["total_parameter_count"] == MODEL_PARAMETERS
    assert loadout["runtime"]["gpu_layers"] == 999
    assert loadout["runtime"]["flash_attention"] is True
    assert loadout["context"]["configured_tokens"] == 8192
    assert loadout["concurrency"]["parallel_slots"] == 4
    assert loadout["speculative_decoding"]["enabled"] is False
    assert loadout["admission"]["admitted"] is False
    assert loadout["loadout_fingerprint"].startswith("sha256:")
    assert loadout["runtime"]["observed_process_argv_sha256"] == (
        "sha256:"
        + hashlib.sha256(
            b"\0".join(item.encode("utf-8") for item in runtime_args(model))
        ).hexdigest()
    )


def test_runtime_rejects_non_loopback_or_partial_gpu(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"x")
    argv = runtime_args(model)
    argv[argv.index("--host") + 1] = "0.0.0.0"
    with pytest.raises(ValueError, match="loopback"):
        parse_runtime_arguments(
            argv,
            model_path=model,
            endpoint="http://127.0.0.1:8000/v1",
        )
    argv = runtime_args(model)
    argv[argv.index("--gpu-layers") + 1] = "20"
    with pytest.raises(ValueError, match="999 GPU-layer"):
        parse_runtime_arguments(
            argv,
            model_path=model,
            endpoint="http://127.0.0.1:8000/v1",
        )


def test_props_reject_runtime_drift(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"x")
    parsed = parse_runtime_arguments(
        runtime_args(model),
        model_path=model,
        endpoint="http://127.0.0.1:8000/v1",
    )
    bad = props(model)
    bad["build_info"] = "wrong-build"
    with pytest.raises(ValueError, match="PrismML build"):
        validate_props(
            bad,
            model_path=model,
            runtime_arguments=parsed,
        )

    bad = props(model)
    bad["default_generation_settings"]["n_ctx"] = 4096
    with pytest.raises(ValueError, match="context"):
        validate_props(
            bad,
            model_path=model,
            runtime_arguments=parsed,
        )

    bad = props(model)
    bad["default_generation_settings"]["params"]["speculative.types"] = "draft"
    with pytest.raises(ValueError, match="speculative"):
        validate_props(
            bad,
            model_path=model,
            runtime_arguments=parsed,
        )
