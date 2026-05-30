from lms_model_fit import analyze_model, estimate_memory_gib, parse_params_b, parse_quant


def test_parse_params_from_model_name():
    assert parse_params_b("qwen3-coder-30b-q4_k_m") == 30.0
    assert parse_params_b("llama-3.1-8B-instruct") == 8.0
    assert parse_params_b("model-without-size") is None


def test_parse_quant_from_model_name():
    quant, bits = parse_quant("qwen3-coder-30b-Q4_K_M.gguf")
    assert quant == "Q4_K_M"
    assert bits > 0
    quant, bits = parse_quant("llama-fp16")
    assert quant == "FP16"
    assert bits == 16.0


def test_estimate_memory():
    estimated = estimate_memory_gib(7.0, 4.5)
    assert estimated is not None
    assert estimated > 0


def test_analyze_model_fit_good_for_small_model():
    profile = {
        "memory": {"mem_total_bytes": 64 * 1024**3, "mem_available_bytes": 48 * 1024**3},
        "gpu": {"nvidia": {"devices": [{"memory.total": "24576"}]}}
    }
    row = analyze_model("tiny-1b-q4", profile)
    assert row["estimated_params_b"] == 1.0
    assert row["fit_grade"] in {"good", "borderline"}
