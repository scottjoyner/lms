from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_runtime_evidence",
    ROOT / "scripts" / "check-runtime-evidence.py",
)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_probe_reports_strong_binding_without_authority(monkeypatch, tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    monkeypatch.setattr(
        module.witness,
        "process_identity",
        lambda _pid: {
            "pid": 42,
            "platform": "darwin",
            "boot_id": "boot",
            "process_start_ticks": 100,
            "process_start_source": "darwin_ps_lstart_epoch_seconds",
            "executable_sha256": "sha256:" + "1" * 64,
            "executable_basename": "runtime",
            "executable_file_identity": {
                "device": 1,
                "inode": 2,
                "size_bytes": 3,
                "mtime_ns": 4,
                "ctime_ns": 5,
            },
        },
    )
    monkeypatch.setattr(
        module.witness,
        "_process_references_model",
        lambda _pid, _path, _identity: "darwin_vmmap_lsof",
    )
    monkeypatch.setattr(module.witness, "_platform_name", lambda: "darwin")

    result = module.probe(42, model)

    assert result["platform"] == "darwin"
    assert result["model_process_binding"] == "darwin_vmmap_lsof"
    assert result["strong_binding"] is True
    assert result["mutating"] is False
    assert result["admission_authority"] is False


def test_probe_keeps_cmdline_only_binding_diagnostic(monkeypatch, tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    monkeypatch.setattr(
        module.witness,
        "process_identity",
        lambda _pid: {
            "pid": 42,
            "platform": "darwin",
            "boot_id": "boot",
            "process_start_ticks": 100,
            "process_start_source": "darwin_ps_lstart_epoch_seconds",
            "executable_sha256": "sha256:" + "1" * 64,
            "executable_basename": "runtime",
            "executable_file_identity": {
                "device": 1,
                "inode": 2,
                "size_bytes": 3,
                "mtime_ns": 4,
                "ctime_ns": 5,
            },
        },
    )
    monkeypatch.setattr(
        module.witness,
        "_process_references_model",
        lambda _pid, _path, _identity: "cmdline",
    )
    monkeypatch.setattr(module.witness, "_platform_name", lambda: "darwin")

    result = module.probe(42, model)

    assert result["model_process_binding"] == "cmdline"
    assert result["strong_binding"] is False
