from argparse import Namespace

import pytest

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


def test_endpoint_maps_must_be_loopback_local():
    assert fleet_bench_entrypoint.parse_endpoint_map(
        ["candidate=http://127.0.0.1:1236/v1"]
    )["candidate"] == "http://127.0.0.1:1236/v1"
    assert fleet_bench_entrypoint.parse_endpoint_map(
        ["candidate=http://localhost:1236"]
    )["candidate"] == "http://localhost:1236/v1"
    with pytest.raises(ValueError, match="loopback"):
        fleet_bench_entrypoint.parse_endpoint_map(
            ["candidate=http://100.64.43.123:1236/v1"]
        )


def test_rollout_script_packages_artifacts_once_and_streams_hashes():
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
    assert "handle.read(8 * 1024 * 1024)" in script
    assert "path.read_bytes()" not in script
    assert script.count('tar -C "$ARTIFACT_DIR"') == 1
    assert 'status=$package_status' in script
    assert "exit \"$status\"" in script
    assert "--dry-run" in script
