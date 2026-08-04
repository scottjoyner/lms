from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from lms_agent_bench import runtime_canary as canary
from lms_agent_bench import runtime_canary_attestation as attestation
from lms_agent_bench import runtime_canary_secure_entrypoint as secure


def command(script: str, *args: str, timeout: float = 5.0):
    return {
        "argv": [str(Path(sys.executable).resolve()), "-c", script, *args],
        "timeout_seconds": timeout,
    }


def event_command(event: str, events: Path, *, fail: bool = False):
    script = (
        "from pathlib import Path; import sys; "
        "p=Path(sys.argv[1]); p.write_text((p.read_text() if p.exists() else '') + sys.argv[2] + '\\n'); "
        + ("raise SystemExit(7)" if fail else "raise SystemExit(0)")
    )
    return command(script, str(events), event)


def probe_command(counter: Path, *, fail_after: int | None = None):
    script = (
        "import json, sys; from pathlib import Path; "
        "p=Path(sys.argv[1]); n=int(p.read_text())+1 if p.exists() else 1; p.write_text(str(n)); "
        "bad=int(sys.argv[2]) > 0 and n >= int(sys.argv[2]); "
        "print(json.dumps({'ok': not bad, 'latency_seconds': 0.01, 'rss_bytes': 1000+n, "
        "'temperature_c': 50.0, 'tps': 20.0, 'ttft_seconds': 0.02})); "
        "raise SystemExit(1 if bad else 0)"
    )
    return command(script, str(counter), str(fail_after or 0))


def make_plan(tmp_path: Path, *, fail_qualification: bool = False, fail_probe_after: int | None = None):
    events = tmp_path / "events.txt"
    counter = tmp_path / "counter.txt"
    commands = {
        "snapshot": event_command("snapshot", events),
        "start_candidate": event_command("start", events),
        "candidate_health": event_command("candidate-health", events),
        "qualification": event_command("qualification", events, fail=fail_qualification),
        "soak_probe": probe_command(counter, fail_after=fail_probe_after),
        "stop_candidate": event_command("stop", events),
        "rollback": event_command("rollback", events),
        "rollback_health": event_command("rollback-health", events),
    }
    plan = {
        "schema_version": canary.PLAN_SCHEMA_VERSION,
        "canary_id": "node-a-candidate-a",
        "loadout_fingerprint": "sha256:" + "a" * 64,
        "working_directory": str(tmp_path),
        "environment_names": [],
        "commands": commands,
        "soak": {
            "duration_seconds": 0.1,
            "interval_seconds": 0.0,
            "minimum_samples": 3,
            "minimum_success_rate": 1.0,
            "max_consecutive_failures": 0,
            "max_p95_latency_seconds": 1.0,
            "max_rss_growth_bytes": 100,
            "minimum_terminal_tps_ratio": 0.9,
            "max_temperature_c": 80.0,
        },
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    path.chmod(0o600)
    return path, events


def test_transaction_runs_qualification_soak_and_verified_rollback(tmp_path):
    plan, events = make_plan(tmp_path)
    workspace = tmp_path / "runs"
    report = canary.execute(plan, workspace, "canary-success", False)
    assert report["success"] is True
    assert report["rollback_succeeded"] is True
    assert report["admission"]["admitted"] is False
    assert events.read_text().splitlines() == [
        "snapshot",
        "start",
        "candidate-health",
        "qualification",
        "stop",
        "rollback",
        "rollback-health",
    ]
    state = json.loads((workspace / "canary-success" / "runtime-canary-state.json").read_text())
    assert state["soak"]["sample_count"] >= 3
    assert state["soak"]["passed"] is True
    assert canary.verify_manifest(workspace / "canary-success", require_success=True)["valid"] is True


def test_failure_after_start_always_attempts_and_verifies_rollback(tmp_path):
    plan, events = make_plan(tmp_path, fail_qualification=True)
    workspace = tmp_path / "runs"
    with pytest.raises(RuntimeError, match="rollback_succeeded=True"):
        canary.execute(plan, workspace, "canary-failed", False)
    assert events.read_text().splitlines() == [
        "snapshot",
        "start",
        "candidate-health",
        "qualification",
        "stop",
        "rollback",
        "rollback-health",
    ]
    report = canary.verify_manifest(workspace / "canary-failed")
    assert report["success"] is False
    assert report["rollback_succeeded"] is True
    with pytest.raises(ValueError, match="did not complete successfully"):
        canary.verify_manifest(workspace / "canary-failed", require_success=True)


def test_soak_failure_triggers_rollback(tmp_path):
    plan, events = make_plan(tmp_path, fail_probe_after=2)
    workspace = tmp_path / "runs"
    with pytest.raises(RuntimeError, match="soak"):
        canary.execute(plan, workspace, "canary-soak-failed", False)
    assert events.read_text().splitlines()[-3:] == ["stop", "rollback", "rollback-health"]
    state = json.loads((workspace / "canary-soak-failed" / "runtime-canary-state.json").read_text())
    assert state["rollback_succeeded"] is True
    assert state["success"] is False


def test_plan_rejects_shell_and_secret_bearing_argv(tmp_path):
    plan, _events = make_plan(tmp_path)
    payload = json.loads(plan.read_text())
    payload["commands"]["qualification"]["argv"] = ["/bin/sh", "-c", "echo unsafe"]
    plan.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="shell interpreter"):
        secure.secure_load_plan(plan)

    payload["commands"]["qualification"]["argv"] = [str(Path(sys.executable).resolve()), "-c", "pass", "--api-key=secret"]
    plan.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="secret-bearing"):
        canary.load_plan(plan)


def test_manifest_tampering_is_rejected(tmp_path):
    plan, _events = make_plan(tmp_path)
    workspace = tmp_path / "runs"
    canary.execute(plan, workspace, "canary-tamper", False)
    state = workspace / "canary-tamper" / "runtime-canary-state.json"
    state.write_bytes(state.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="size mismatch|SHA-256 mismatch"):
        canary.verify_manifest(workspace / "canary-tamper")


def test_soak_summary_catches_leak_throttle_and_heat():
    samples = [
        {"ok": True, "latency_seconds": 0.1, "rss_bytes": 1000, "tps": 20, "temperature_c": 50},
        {"ok": True, "latency_seconds": 0.1, "rss_bytes": 1200, "tps": 20, "temperature_c": 60},
        {"ok": True, "latency_seconds": 0.1, "rss_bytes": 2000, "tps": 10, "temperature_c": 90},
    ]
    policy = {
        "minimum_samples": 3,
        "minimum_success_rate": 1.0,
        "max_consecutive_failures": 0,
        "max_p95_latency_seconds": 1.0,
        "max_rss_growth_bytes": 500,
        "minimum_terminal_tps_ratio": 0.8,
        "max_temperature_c": 80,
    }
    summary = canary.summarize_soak(samples, policy)
    assert summary["passed"] is False
    assert "rss_growth" in summary["failures"]
    assert "temperature" in summary["failures"]


pytestmark_attestation = pytest.mark.skipif(
    shutil.which("ssh-keygen") is None,
    reason="OpenSSH ssh-keygen is unavailable",
)


@pytestmark_attestation
def test_canary_manifest_can_be_signed_and_verified(tmp_path):
    plan, _events = make_plan(tmp_path)
    workspace = tmp_path / "runs"
    canary.execute(plan, workspace, "canary-signed", False)
    root = workspace / "canary-signed"
    key = tmp_path / "canary-key"
    generated = subprocess.run(
        ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)],
        capture_output=True,
        check=False,
    )
    assert generated.returncode == 0
    key.chmod(0o600)
    allowed = tmp_path / "allowed_signers"
    allowed.write_text(
        "runtime-canary-operator " + key.with_suffix(".pub").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    attestation.sign_run(root, key, require_success=True)
    verified = attestation.verify_attestation(
        root,
        allowed,
        "runtime-canary-operator",
        require_success=True,
    )
    assert verified["valid"] is True
    assert verified["rollback_succeeded"] is True
    assert verified["admission"]["admitted"] is False
