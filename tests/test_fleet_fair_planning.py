from lms_agent_bench.fleet_loadout_entrypoint import (
    build_plan,
    kv_bytes_per_token,
)


def observation(total_bytes=32 * 1024**3):
    return {
        "observation_fingerprint": "sha256:observation",
        "hardware": {
            "memory": {"total_bytes": total_bytes},
            "cpu": {"logical_processors": 8, "physical_processors": 4},
            "accelerators": {},
            "supported_backends": ["vulkan", "cpu"],
        },
    }


def test_candidate_limit_is_fair_across_models_and_backends():
    models = [
        {
            "id": "alpha-7B-Q4.gguf",
            "path": "/models/alpha.gguf",
            "size_bytes": 4 * 1024**3,
            "parameter_billions": 7,
            "max_context": 8192,
        },
        {
            "id": "beta-9B-Q4.gguf",
            "path": "/models/beta.gguf",
            "size_bytes": 5 * 1024**3,
            "parameter_billions": 9,
            "max_context": 8192,
        },
    ]
    plan = build_plan(observation(), models, contexts=[4096], max_candidates=4)
    groups = {
        (item["model"]["id"], item["backend"])
        for item in plan["candidates"]
    }
    assert groups == {
        ("alpha-7B-Q4.gguf", "vulkan"),
        ("alpha-7B-Q4.gguf", "cpu"),
        ("beta-9B-Q4.gguf", "vulkan"),
        ("beta-9B-Q4.gguf", "cpu"),
    }
    assert plan["planning_policy"]["truncation"] == "round_robin_by_model_and_backend"


def test_default_kv_estimate_keeps_viable_35b_8k_candidate():
    model = {
        "id": "qwen-35B-Q4.gguf",
        "path": "/models/qwen.gguf",
        "size_bytes": 20 * 1024**3,
        "parameter_billions": 35,
        "max_context": 8192,
    }
    assert kv_bytes_per_token(model) < 512 * 1024
    plan = build_plan(observation(), [model], contexts=[8192], max_candidates=8)
    assert any(
        item["model"]["id"] == model["id"] and item["context_tokens"] == 8192
        for item in plan["candidates"]
    )


def test_explicit_architecture_metadata_controls_kv_estimate():
    model = {
        "num_hidden_layers": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "kv_element_bytes": 2,
    }
    assert kv_bytes_per_token(model) == 131072
