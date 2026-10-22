"""Synthetic safety and hierarchical-cgroup tests for Joyner memory triage."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/r9700_memory_attribution.py"
spec = importlib.util.spec_from_file_location("r9700_memory_attribution", MODULE_PATH)
assert spec and spec.loader
triage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(triage)


@pytest.fixture
def trees(tmp_path):
    proc = tmp_path / "proc"
    cg = tmp_path / "cg"
    (proc / "pressure").mkdir(parents=True)
    cg.mkdir()
    (proc / "meminfo").write_text(
        "MemTotal:       15470664 kB\n"
        "MemAvailable:   3090732 kB\n"
        "Shmem:         2281208 kB\n"
        "SwapTotal:     4194300 kB\n"
        "SwapFree:      1179928 kB\n",
    )
    (proc / "pressure/memory").write_text(
        "some avg10=0.00 avg60=0.00 avg300=0.01 total=12\n"
        "full avg10=0.00 avg60=0.00 avg300=0.00 total=9\n",
    )
    def group(name, shmem, anon=0, current=123000000):
        p = cg / name
        p.mkdir(parents=True, exist_ok=True)
        (p / "memory.stat").write_text(
            f"shmem {shmem}\nanon {anon}\nfile 3500000000\n",
        )
        (p / "memory.current").write_text(f"{current}\n")
    group(".", 2281208 * 1024)
    group("user.slice", 637 * 2**20, 200 * 2**20)
    group("user.slice/user-1001.slice", 637 * 2**20)
    group("user.slice/user-1001.slice/joyner-lane.service", 637 * 2**20)
    group("system.slice", 36 * 2**20)
    group("init.scope", 1 * 2**20)
    pid = proc / "1400830"
    pid.mkdir()
    (pid / "comm").write_text("llama-server\n")
    (pid / "status").write_text(
        "Name:\tllama-server\nVmRSS: 87000 kB\nVmSwap: 88000 kB\n",
    )
    (pid / "smaps_rollup").write_text(
        "Rss: 87000 kB\nPss_Shmem: 0 kB\n",
    )
    return proc, cg


def run(trees):
    proc, cg = trees
    return triage.collect(proc_root=proc, cgroup_root=cg, target_host="joyner")


def test_fails_closed_when_running_on_another_host(trees):
    proc, cg = trees
    result = triage.collect(proc_root=proc, cgroup_root=cg, target_host="x1-370")
    assert result["status"] == "HOST_MISMATCH"
    assert result["production_admission"] == "NO_GO"
    assert not result["model_launch_allowed"]


def test_correct_read_only_memory_gates(trees):
    result = run(trees)
    assert result["status"] == "READ_ONLY_OBSERVATION"
    assert result["memory_capacity_gate"] == "HOLD"
    assert result["available_memory_gate"] == "HOLD"
    assert result["observed_memory"]["installed_gib"] < 16
    assert result["observed_memory"]["available_gib"] < 4
    assert result["production_admission"] == "NO_GO"
    assert result["full_model_benchmark_executed"] is False
    assert result["gpu_loaded"] is False


def test_no_double_counting_nested_user_slice(trees):
    result = run(trees)
    c = result["cgroup_shared_memory"]
    root = 2281208 * 1024
    expected = root - (637 + 36 + 1) * 2**20
    assert c["top_level_unattributed_mib"] == round(expected / 2**20, 2)
    assert c["root_available"] is True
    assert c["sampled_cgroup_count"] == 6
    assert sum(x["shmem_mib"] for x in c["top_consumers"]) > 637
    assert "not a measured leak" in c["unattributed_interpretation"]


def test_one_process_rss_does_not_get_confused_with_cgroup_shared(trees):
    result = run(trees)
    row = result["existing_llama_server_processes"][0]
    assert row["pid"] == 1400830
    assert row["vmrss_mib"] == round(87000 / 1024, 2)
    assert row["pss_shmem_mib"] == 0
    assert result["cgroup_shared_memory"]["top_consumers"][0]["shmem_mib"] > 2000


def test_permission_denied_smaps_does_not_claim_zero(trees):
    proc, cg = trees
    (proc / "1400830/smaps_rollup").unlink()
    result = run(trees)
    row = result["existing_llama_server_processes"][0]
    assert row["smaps_readable"] is False
    assert row["pss_shmem_mib"] is None


def test_pid_process_dies_after_scan_is_safe(trees):
    proc, cg = trees
    (proc / "1400830/status").unlink()
    (proc / "1400830/smaps_rollup").unlink()
    row = run(trees)["existing_llama_server_processes"][0]
    assert row["vmrss_mib"] is None
    assert row["vmswap_mib"] is None


def test_absent_kernel_memory_fails_closed(trees):
    proc, cg = trees
    (proc / "meminfo").unlink()
    result = run(trees)
    assert result["status"] == "INCOMPLETE"
    assert result["production_admission"] == "NO_GO"


def test_inconsistent_kernel_swap_fails_closed(trees):
    proc, cg = trees
    with (proc / "meminfo").open("a") as f:
        f.write("SwapFree: 999999999 kB\n")
    result = run(trees)
    assert result["status"] == "INCOMPLETE"


def test_psi_missing_fails_closed(trees):
    proc, cg = trees
    (proc / "pressure/memory").write_text("")
    result = run(trees)
    assert result["status"] == "INCOMPLETE"


def test_missing_cgroup_hierarchy_does_not_fake_attribution(trees):
    proc, cg = trees
    result = triage.collect(
        proc_root=proc, cgroup_root=cg / "no-such-dir", target_host="joyner",
    )
    assert result["status"] == "READ_ONLY_OBSERVATION"
    assert result["cgroup_shared_memory"]["root_available"] is False
    assert result["cgroup_shared_memory"]["top_level_unattributed_mib"] is None
    assert result["production_admission"] == "NO_GO"


def test_psi_extracted_without_total_counter(trees):
    result = run(trees)
    assert result["observed_memory"]["memory_psi"]["some"]["avg300"] == .01
    assert result["observed_memory"]["memory_psi"]["full"]["avg10"] == .0


def test_truncated_file_does_not_produce_partial_status(trees):
    proc, cg = trees
    (proc / "meminfo").write_bytes(b"X" * 257000)
    result = run(trees)
    assert result["status"] == "INCOMPLETE"


def test_host_override_is_only_test_injection(trees):
    proc, cg = trees
    assert run(trees)["host_observed"] == "joyner"
    assert triage.collect(
        proc_root=proc, cgroup_root=cg, target_host="other-host"
    )["status"] == "HOST_MISMATCH"


def test_disk_fixture_contents_remain_unchanged(trees):
    proc, cg = trees
    before = (proc / "meminfo").read_bytes()
    cg_before = (cg / "memory.stat").read_bytes()
    run(trees)
    assert (proc / "meminfo").read_bytes() == before
    assert (cg / "memory.stat").read_bytes() == cg_before


def test_safe_action_does_not_recommend_restart_or_swap_clear(trees):
    result = run(trees)
    action = result["safe_next_action"]
    assert "Do not clear swap" in action
    assert "64 GiB RAM upgrade" in action
    assert not result["model_launch_allowed"]