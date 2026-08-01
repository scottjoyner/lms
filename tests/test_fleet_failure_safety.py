from argparse import Namespace
from pathlib import Path

from lms_agent_bench import fleet_bench_entrypoint
from lms_agent_bench import fleet_rollout_entrypoint


def test_unmapped_non_llama_dry_run_is_rendered_not_failed(tmp_path):
    candidate = {
        "candidate_id": "npu-candidate",
        "engine": "npu-inference-server",
        "backend": "npu_xdna2",
        "model": {"id": "npu-model"},
    }
    args = Namespace(
        dry_run=True,
        output_dir=str(tmp_path),
        llama_server_bin=None,
    )
    result = fleet_bench_entrypoint.execute_candidate(candidate, args, {}, {})
    assert result["benchmark_exit_code"] == 0
    assert result["requires_endpoint_map"] is True
    assert result["launch_mode"] == "existing_or_adapter"
    assert (tmp_path / "npu-candidate" / "result.json").exists()


def test_llama_dry_run_does_not_require_installed_binary(tmp_path):
    model = tmp_path / "model-Q4_K_M.gguf"
    model.write_bytes(b"model")
    candidate = {
        "candidate_id": "llama-candidate",
        "engine": "llama.cpp",
        "backend": "cpu",
        "model": {"id": model.name, "path": str(model)},
        "context_tokens": 4096,
        "parallel_slots": 1,
        "threads": 2,
        "gpu_layers": 0,
        "flash_attention": False,
        "batch_size": 256,
        "ubatch_size": 128,
        "benchmark_port": 18080,
    }
    args = Namespace(
        dry_run=True,
        output_dir=str(tmp_path),
        llama_server_bin="definitely-not-installed-llama-server",
    )
    result = fleet_bench_entrypoint.execute_candidate(candidate, args, {}, {})
    assert result["benchmark_exit_code"] == 0
    assert result["binary_available"] is False
    assert result["launch_command"][0] == "definitely-not-installed-llama-server"


def test_rollout_script_packages_artifacts_on_exit():
    node = {
        "node_id": "x1-370",
        "ssh_target": "scott@x1-370",
        "repo_dir": "/home/scott/git/lms",
        "branch": "full-auto-reconciliation-20260730",
        "model_roots": ["/models"],
        "contexts": [4096, 8192],
    }
    script = fleet_rollout_entrypoint.build_remote_script(node, "run-1")
    assert "trap lms_fleet_package_artifacts EXIT" in script
    assert "remote_exit_code" in script
    assert "rm -f \"$ARTIFACT_DIR/bundle_manifest.json\"" in script
    assert "exit \"$status\"" in script
    assert "--dry-run" in script
