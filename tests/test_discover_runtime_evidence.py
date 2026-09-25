from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "discover_runtime_evidence",
    ROOT / "scripts" / "discover-runtime-evidence.py",
)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_candidate_processes_parses_each_pid_independently(monkeypatch):
    monkeypatch.setattr(
        module,
        "_run",
        lambda *_args, **_kwargs: (
            " 10060     1 /Applications/LM Studio.app/Contents/MacOS/LM Studio\n"
            " 10293 10060 /Users/me/.cache/lm-studio/node llmworker.js\n"
            "   896     1 python -m assistx.fleet_node_agent\n"
        ),
    )

    processes = module.candidate_processes()

    assert [item["pid"] for item in processes] == [10060, 10293]
    assert processes[1]["ppid"] == 10060


def test_open_weight_paths_handles_spaces_without_shell_splitting(
    monkeypatch, tmp_path
):
    model_dir = tmp_path / "Model With Spaces"
    model_dir.mkdir()
    shard = model_dir / "model-00001-of-00002.safetensors"
    shard.write_bytes(b"weights")

    monkeypatch.setattr(
        module,
        "_run",
        lambda *_args, **_kwargs: (
            "p10293\n"
            f"n{shard}\n"
            "n/tmp/not-a-weight.json\n"
        ),
    )

    assert module.open_weight_paths(10293) == [shard]


def test_inspect_process_marks_matching_darwin_binding_strong(
    monkeypatch, tmp_path
):
    shard = tmp_path / "model.safetensors"
    shard.write_bytes(b"weights")
    monkeypatch.setattr(module, "open_weight_paths", lambda _pid: [shard])
    monkeypatch.setattr(module, "vmmap_weight_paths", lambda _pid: [str(shard)])
    monkeypatch.setattr(
        module.witness,
        "process_identity",
        lambda pid: {"pid": pid, "platform": "darwin"},
    )
    monkeypatch.setattr(
        module.witness,
        "_stat_identity",
        lambda _stat: {
            "device": 1,
            "inode": 2,
            "size_bytes": 7,
            "mtime_ns": 8,
            "ctime_ns": 9,
        },
    )
    monkeypatch.setattr(
        module.witness,
        "_process_references_model",
        lambda _pid, _path, _identity: "darwin_vmmap_lsof",
    )

    result = module.inspect_process(10293)

    assert result["open_weight_paths"] == [str(shard)]
    assert result["probes"][0]["status"] == "strong"
    assert result["probes"][0]["strong_binding"] is True
    assert result["probes"][0]["model_process_binding"] == "darwin_vmmap_lsof"
