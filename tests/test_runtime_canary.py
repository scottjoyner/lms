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


def test_command_executable_drift_refuses_dispatch(tmp_path):
    script = tmp_path / "stage"
    marker = tmp_path / "executed"
    script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    script.chmod(0o700)
    spec = canary._command({"argv": [str(script)], "timeout_seconds": 5}, "snapshot")
    script.write_text("#!/bin/sh\ntouch " + str(marker) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="executable .*changed"):
        canary.run_command("snapshot", spec, cwd=tmp_path, env={}, logs=tmp_path)
    assert not marker.exists()


def test_soak_probe_executable_drift_refuses_dispatch(tmp_path):
    script = tmp_path / "probe"
    marker = tmp_path / "executed"
    script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    script.chmod(0o700)
    spec = canary._command({"argv": [str(script)], "timeout_seconds": 5}, "soak_probe")
    script.write_text("#!/bin/sh\ntouch " + str(marker) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="executable .*changed"):
        canary._captured_probe(spec, cwd=tmp_path, env={})
    assert not marker.exists()


def test_canary_symlink_retarget_refuses_dispatch(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    alias = tmp_path / "stage"
    marker = tmp_path / "executed"
    a.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    b.write_text("#!/bin/sh\ntouch " + str(marker) + "\n", encoding="utf-8")
    a.chmod(0o700)
    b.chmod(0o700)
    alias.symlink_to(a)
    spec = canary._command({"argv": [str(alias)], "timeout_seconds": 5}, "snapshot")
    alias.unlink()
    alias.symlink_to(b)
    with pytest.raises(ValueError, match="executable .*changed"):
        canary.run_command("snapshot", spec, cwd=tmp_path, env={}, logs=tmp_path)
    assert not marker.exists()


def test_failed_candidate_start_triggers_rollback(tmp_path):
    plan, events = make_plan(tmp_path)
    payload = json.loads(plan.read_text())
    payload["commands"]["start_candidate"] = event_command("start-partial", events, fail=True)
    plan.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="rollback_succeeded=True"):
        canary.execute(plan, tmp_path / "runs", "failed-start", False)
    assert events.read_text().splitlines() == [
        "snapshot", "start-partial", "stop", "rollback", "rollback-health"
    ]
    root = tmp_path / "runs" / "failed-start"
    state = json.loads((root / "runtime-canary-state.json").read_text())
    assert state["candidate_started"] is False
    assert state["candidate_start_attempted"] is True
    assert state["rollback_attempted"] is True
    assert state["rollback_succeeded"] is True
    assert state["success"] is False
    assert canary.verify_manifest(root)["valid"] is True
    with pytest.raises(ValueError, match="did not complete successfully"):
        canary.verify_manifest(root, require_success=True)


def test_timed_out_candidate_start_triggers_rollback(tmp_path):
    plan, events = make_plan(tmp_path)
    payload = json.loads(plan.read_text())
    script = ("from pathlib import Path;import time,sys;"
              "p=Path(sys.argv[1]);p.write_text(p.read_text()+'start-timeout\\n');"
              "time.sleep(10)")
    payload["commands"]["start_candidate"] = command(script, str(events), timeout=0.1)
    plan.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="rollback_succeeded=True"):
        canary.execute(plan, tmp_path / "runs", "timeout-start", False)
    assert events.read_text().splitlines() == [
        "snapshot", "start-timeout", "stop", "rollback", "rollback-health"
    ]
    root = tmp_path / "runs" / "timeout-start"
    state = json.loads((root / "runtime-canary-state.json").read_text())
    assert state["candidate_start_attempted"] is True
    assert state["rollback_succeeded"] is True
    assert state["success"] is False


def test_snapshot_failure_does_not_invoke_candidate_recovery(tmp_path):
    plan, events = make_plan(tmp_path)
    payload = json.loads(plan.read_text())
    payload["commands"]["snapshot"] = event_command("snapshot-failed", events, fail=True)
    plan.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="rollback_succeeded=False"):
        canary.execute(plan, tmp_path / "runs", "failed-snapshot", False)
    assert events.read_text().splitlines() == ["snapshot-failed"]
    root = tmp_path / "runs" / "failed-snapshot"
    state = json.loads((root / "runtime-canary-state.json").read_text())
    assert state["candidate_started"] is False
    assert state.get("candidate_start_attempted") is not True
    assert state["rollback_attempted"] is False
    assert state["success"] is False


def test_timed_out_soak_probe_terminates_descendant(tmp_path):
    import time

    marker = tmp_path / "late-write"
    child = (
        "import sys,time;from pathlib import Path;"
        "time.sleep(0.65);Path(sys.argv[1]).write_text('descendant-survived')"
    )
    parent = (
        "import subprocess,sys,time;"
        "subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]]);"
        "time.sleep(5)"
    )
    spec = canary._command(command(parent, child, str(marker), timeout=0.12), "soak_probe")
    sample = canary._captured_probe(spec, cwd=tmp_path, env={})
    assert sample["timed_out"] is True
    assert sample["ok"] is False
    time.sleep(0.85)
    assert not marker.exists(), "timed-out soak probe left a live child process"


def test_probe_parent_exits_descendant_keeps_pipe(tmp_path):
    import time

    marker = tmp_path / "late-write-after-parent-exit"
    child = (
        "import sys,time;from pathlib import Path;"
        "time.sleep(0.65);Path(sys.argv[1]).write_text('descendant-survived')"
    )
    # The child inherits the parent's stdout pipe. The parent exits first.
    parent = (
        "import subprocess,sys;"
        "subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]])"
    )
    spec = canary._command(command(parent, child, str(marker), timeout=0.12), "soak_probe")
    sample = canary._captured_probe(spec, cwd=tmp_path, env={})
    assert sample["timed_out"] is True
    assert sample["ok"] is False
    time.sleep(0.85)
    assert not marker.exists(), "parent exit left a soak probe descendant running"


@pytest.mark.parametrize("parent_exits", [False, True])
def test_probe_timeout_kills_sigterm_resistant_child(tmp_path, parent_exits):
    import time

    late = tmp_path / "late-sigterm-ignored"
    ready = tmp_path / "child-ready"
    child = (
        "import signal,sys,time;from pathlib import Path;"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        "Path(sys.argv[1]).write_text('ready');"
        "time.sleep(0.7);Path(sys.argv[2]).write_text('child-survived')"
    )
    # The parent waits for the child's ready marker so SIGTERM cannot beat
    # the child's signal handler registration and make the test vacuous.
    parent = (
        "import subprocess,sys,time;from pathlib import Path;"
        "subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2],sys.argv[3]]);"
        "ready=Path(sys.argv[2]);deadline=time.monotonic()+0.4;"
        "exec('while not ready.exists() and time.monotonic()<deadline: time.sleep(0.005)')"
    ) + ("" if parent_exits else ";time.sleep(5)")
    spec = canary._command(
        command(parent, child, str(ready), str(late), timeout=0.16), "soak_probe"
    )
    sample = canary._captured_probe(spec, cwd=tmp_path, env={})
    assert ready.exists(), "SIGTERM-ignore handler was not established"
    assert sample["timed_out"] is True and sample["ok"] is False
    time.sleep(0.82)
    assert not late.exists(), "SIGTERM-ignoring probe descendant survived timeout"


@pytest.mark.parametrize("failed_stage", ["stop_candidate", "rollback"])
def test_recovery_command_failure_does_not_claim_rollback(tmp_path, failed_stage):
    # Recovery health may succeed without proving that stop or restore did.
    plan, events = make_plan(tmp_path, fail_qualification=True)
    payload = json.loads(plan.read_text())
    payload["commands"][failed_stage] = event_command(failed_stage + "-failed", events, fail=True)
    plan.write_text(json.dumps(payload))
    workspace = tmp_path / "runs"
    with pytest.raises(RuntimeError, match="rollback_succeeded=False"):
        canary.execute(plan, workspace, "recovery-unsafe", False)
    state = json.loads((workspace / "recovery-unsafe" / "runtime-canary-state.json").read_text())
    assert state["success"] is False
    assert state["stage"] == "rollback_failed"
    assert state["rollback_attempted"] is True
    assert state["rollback_succeeded"] is False
    assert [x["command"] for x in state["recovery_failures"]] == [failed_stage]
    assert events.read_text().splitlines()[-3:] == [
        "stop_candidate-failed" if failed_stage == "stop_candidate" else "stop",
        "rollback-failed" if failed_stage == "rollback" else "rollback",
        "rollback-health",
    ]
    report = canary.verify_manifest(workspace / "recovery-unsafe")
    assert report["valid"] and not report["rollback_succeeded"]
    with pytest.raises(ValueError, match="did not complete successfully"):
        canary.verify_manifest(workspace / "recovery-unsafe", require_success=True)


def _reseal_structural_manifest_for_negative_test(root: Path):
    # Deliberately forge an UNSIGNED local manifest to test structural checks.
    manifest_path = root / "runtime-canary-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest["artifacts"]:
        path = root / item["path"]
        item["size_bytes"] = path.stat().st_size
        item["sha256"] = canary._operator.file_sha256(path)
    core = {key: value for key, value in manifest.items() if key not in {"created_at_utc", "manifest_fingerprint"}}
    manifest["manifest_fingerprint"] = canary._operator.canonical_hash(core)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_rehashed_manifest_missing_soak_evidence_rejected(tmp_path):
    plan, _ = make_plan(tmp_path)
    workspace = tmp_path / "runs"
    canary.execute(plan, workspace, "missing-soak", False)
    root = workspace / "missing-soak"
    (root / "soak-samples.jsonl").unlink()
    manifest_path = root / "runtime-canary-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"] = [x for x in manifest["artifacts"] if x["path"] != "soak-samples.jsonl"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _reseal_structural_manifest_for_negative_test(root)
    with pytest.raises(ValueError, match="missing|soak|incomplete"):
        canary.verify_manifest(root, require_success=True)


def test_rehashed_manifest_with_empty_command_history_rejected(tmp_path):
    plan, _ = make_plan(tmp_path)
    workspace = tmp_path / "runs"
    canary.execute(plan, workspace, "missing-commands", False)
    root = workspace / "missing-commands"
    state_path = root / "runtime-canary-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["commands"] = []
    state_path.write_text(json.dumps(state), encoding="utf-8")
    _reseal_structural_manifest_for_negative_test(root)
    with pytest.raises(ValueError, match="commands|evidence|incomplete"):
        canary.verify_manifest(root, require_success=True)


def test_rehashed_soak_samples_contradict_success_summary_rejected(tmp_path):
    plan, _ = make_plan(tmp_path)
    workspace = tmp_path / "runs"
    canary.execute(plan, workspace, "forged-soak", False)
    root = workspace / "forged-soak"
    samples_path = root / "soak-samples.jsonl"
    rows = [json.loads(line) for line in samples_path.read_text(encoding="utf-8").splitlines()]
    assert rows and rows[0]["ok"] is True
    rows[0]["ok"] = False
    samples_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    # Update the state-stored digest as well, as a locally controlled attacker could.
    state_path = root / "runtime-canary-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["soak"]["samples_sha256"] = canary._operator.file_sha256(samples_path)
    state_path.write_text(json.dumps(state), encoding="utf-8")
    _reseal_structural_manifest_for_negative_test(root)
    with pytest.raises(ValueError, match="soak|sample|inconsistent"):
        canary.verify_manifest(root, require_success=True)


@pytest.mark.parametrize(
    "field",
    [
        "argv",
        "argv_missing",
        "executable_resolved",
        "executable_sha256",
        "timeout_seconds",
        "timeout_seconds_type",
    ],
)
def test_rehashed_success_command_must_match_authorized_plan(tmp_path, field):
    # Recalculate local UNSIGNED manifest hashes after altering the receipt.
    plan, _ = make_plan(tmp_path)
    workspace = tmp_path / "runs"
    canary.execute(plan, workspace, "forged-command", False)
    root = workspace / "forged-command"
    state_path = root / "runtime-canary-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    record = state["commands"][1]
    if field == "argv":
        record["argv"] = [*record["argv"], "--unapproved"]
    elif field == "argv_missing":
        del record["argv"]
    elif field == "executable_resolved":
        record["executable_resolved"] += "-unapproved"
    elif field == "executable_sha256":
        record["executable_sha256"] = "sha256:" + "0" * 64
    elif field == "timeout_seconds":
        record["timeout_seconds"] += 2.0
    elif field == "timeout_seconds_type":
        record["timeout_seconds"] = True
    state_path.write_text(json.dumps(state), encoding="utf-8")
    _reseal_structural_manifest_for_negative_test(root)
    with pytest.raises(ValueError, match="does not match normalized plan"):
        canary.verify_manifest(root, require_success=True)
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tps", None),
        ("rss_bytes", None),
        ("latency_seconds", None),
        ("tps", 0.0),
        ("rss_bytes", -1.0),
        ("latency_seconds", -0.5),
        ("tps", float("nan")),
    ],
)
def test_successful_soak_requires_valid_measurements(field, value):
    policy = {
        "minimum_samples": 3,
        "minimum_success_rate": 1.0,
        "max_consecutive_failures": 0,
        "max_p95_latency_seconds": 1.0,
        "max_rss_growth_bytes": 500,
        "minimum_terminal_tps_ratio": 0.8,
        "max_temperature_c": 80.0,
    }
    samples = [
        {"ok": True, "latency_seconds": 0.1,
         "rss_bytes": 1024.0, "tps": 20.0, "temperature_c": 50.0}
        for _ in range(3)
    ]
    if value is None:
        del samples[1][field]
    else:
        samples[1][field] = value
    summary = canary.summarize_soak(samples, policy)
    assert summary["passed"] is False
    assert f"invalid_{field}" in summary["failures"]


@pytest.mark.parametrize(
    ("field", "literal"),
    [
        ("tps", "'tps': 20.0, "),
        ("rss_bytes", "'rss_bytes': 1000+n, "),
    ],
)
def test_real_canary_soak_missing_required_metric_triggers_rollback(tmp_path, field, literal):
    plan, events = make_plan(tmp_path)
    raw = json.loads(plan.read_text(encoding="utf-8"))
    argv = raw["commands"]["soak_probe"]["argv"]
    assert argv[2].count(literal) == 1
    argv[2] = argv[2].replace(literal, "")
    plan.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(RuntimeError, match="soak gate failed"):
        canary.execute(plan, tmp_path / "runs", f"missing-{field}", False)
    assert events.read_text().splitlines()[-3:] == [
        "stop", "rollback", "rollback-health",
    ]
    report = canary.verify_manifest(tmp_path / "runs" / f"missing-{field}")
    assert report["success"] is False
    assert report["rollback_succeeded"] is True
    assert report["admission"]["admitted"] is False
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("returncode", 7),
        ("returncode", False),
        ("timed_out", True),
        ("timed_out", 0),
        ("parse_error", "unexpected parse failure"),
        ("captured_at_utc", "not-a-timestamp"),
        ("duration_seconds", -1.0),
    ],
)
def test_rehashed_success_soak_rejects_contradictory_probe_receipt(tmp_path, field, value):
    # A locally rehashed unsigned manifest must not turn a failed or
    # malformed probe execution receipt into a successful observation.
    plan, _ = make_plan(tmp_path)
    workspace = tmp_path / "runs"
    canary.execute(plan, workspace, "forged-probe-status", False)
    root = workspace / "forged-probe-status"
    samples_path = root / "soak-samples.jsonl"
    rows = [json.loads(line) for line in samples_path.read_text(encoding="utf-8").splitlines()]
    assert rows and rows[0]["ok"] is True
    rows[0][field] = value
    samples_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    state_path = root / "runtime-canary-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["soak"]["samples_sha256"] = canary._operator.file_sha256(samples_path)
    state_path.write_text(json.dumps(state), encoding="utf-8")
    _reseal_structural_manifest_for_negative_test(root)
    with pytest.raises(ValueError, match="soak sample"):
        canary.verify_manifest(root, require_success=True)
@pytest.mark.parametrize("failing_step", ["state_write", "manifest_write"])
def test_artifact_seal_failure_releases_runtime_canary_lock(tmp_path, monkeypatch, failing_step):
    plan, events = make_plan(tmp_path)
    workspace = tmp_path / "runs"
    if failing_step == "state_write":
        original = canary._write_state

        def failing_state(path, state):
            if state.get("completed_at_utc") is not None:
                raise OSError("injected final state write failure")
            return original(path, state)

        monkeypatch.setattr(canary, "_write_state", failing_state)
    else:
        def failing_manifest(_root, _state):
            raise OSError("injected manifest seal failure")

        monkeypatch.setattr(canary, "build_manifest", failing_manifest)

    with pytest.raises(OSError, match="injected"):
        canary.execute(plan, workspace, "seal-io-failure", False)
    assert events.read_text().splitlines()[-3:] == [
        "stop", "rollback", "rollback-health",
    ]
    assert not (workspace / ".runtime-canary.lock").exists()


def test_lock_owner_write_failure_removes_unowned_lock(tmp_path, monkeypatch):
    plan, _events = make_plan(tmp_path)
    workspace = tmp_path / "runs"
    original = canary._operator.write_json

    def failing_owner(path, content):
        if Path(path).name == "owner.json":
            raise OSError("injected lock owner write failure")
        return original(path, content)

    monkeypatch.setattr(canary._operator, "write_json", failing_owner)
    with pytest.raises(RuntimeError, match="injected lock owner"):
        canary.execute(plan, workspace, "lock-owner-io-failure", False)
    assert not (workspace / ".runtime-canary.lock").exists()
    assert not (workspace / "lock-owner-io-failure").exists()


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_oversize_soak_probe_output_is_stopped_before_completion(tmp_path, stream):
    # An 8MiB flood is harmless in the test, but an unlimited communicate()
    # reads the entire stream before applying the claimed one-MiB limit.
    finished = tmp_path / "probe_finished"
    script = (
        "import sys;from pathlib import Path;"
        "out=getattr(sys,sys.argv[1]).buffer;"
        "out.write(b'x'*(8*1024*1024));out.flush();"
        "Path(sys.argv[2]).write_text('completed')"
    )
    spec = canary._command(command(script, stream, str(finished), timeout=3), "soak_probe")
    with pytest.raises(RuntimeError, match="soak probe output exceeded one MiB"):
        canary._captured_probe(spec, cwd=tmp_path, env={})
    assert not finished.exists(), "oversized probe was allowed to finish flooding output"

@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_lifecycle_log_flood_is_stopped_before_command_finishes(tmp_path, stream):
    finished = tmp_path / "flood-completed"
    code = (
        "import sys; from pathlib import Path;"
        "stream=getattr(sys,sys.argv[1]).buffer;"
        "stream.write(b'x'*(16*1024*1024));stream.flush();"
        "Path(sys.argv[2]).write_text('finished')"
    )
    spec = canary._command(command(code, stream, str(finished), timeout=3), "snapshot")
    with pytest.raises(RuntimeError, match="lifecycle command .* output exceeded"):
        canary.run_command("snapshot", spec, cwd=tmp_path, env={}, logs=tmp_path)
    assert not finished.exists(), "lifecycle command completed after exceeding output budget"
    assert (tmp_path / "snapshot.stdout.log").stat().st_size <= canary._MAX_COMMAND_LOG_BYTES
    assert (tmp_path / "snapshot.stderr.log").stat().st_size <= canary._MAX_COMMAND_LOG_BYTES
@pytest.mark.parametrize("scenario", ["sample_count", "evidence_bytes"])
def test_soak_producer_limits_apply_before_unbounded_collection(tmp_path, monkeypatch, scenario):
    plan, _events = make_plan(tmp_path)
    normalized = canary.load_plan(plan)
    policy = dict(normalized["soak"])
    policy["duration_seconds"] = 30.0
    policy["minimum_samples"] = 1
    policy["interval_seconds"] = 0.0
    output = tmp_path / f"bounded-{scenario}.jsonl"
    calls = []

    def synthetic_probe(_spec, _cwd, _env):
        calls.append(1)
        if len(calls) > (3 if scenario == "sample_count" else 1):
            raise AssertionError("probe called after evidence budget should have failed")
        sample = {
            "ok": True, "returncode": 0, "timed_out": False,
            "parse_error": None, "duration_seconds": 0.001,
            "captured_at_utc": canary.utc_now(),
            "latency_seconds": 0.01, "rss_bytes": 1024,
            "tps": 20.0, "temperature_c": 50.0,
            "stderr_tail": "",
        }
        if scenario == "evidence_bytes":
            sample["stderr_tail"] = "x" * 600
        return sample

    monkeypatch.setattr(canary, "_captured_probe", synthetic_probe)
    if scenario == "sample_count":
        monkeypatch.setattr(canary, "_MAX_SOAK_SAMPLES", 3)
        with pytest.raises(RuntimeError, match="soak sample count limit"):
            canary.run_soak(normalized["commands"]["soak_probe"], policy,
                           cwd=tmp_path, env={}, output=output)
        assert len(calls) == 3
        assert len(output.read_text().splitlines()) == 3
    else:
        monkeypatch.setattr(canary, "_MAX_SOAK_BYTES", 500)
        with pytest.raises(RuntimeError, match="soak evidence size limit"):
            canary.run_soak(normalized["commands"]["soak_probe"], policy,
                           cwd=tmp_path, env={}, output=output)
        assert len(calls) == 1
        assert not output.exists() or output.stat().st_size <= 500


def test_plan_rejects_impossible_soak_minimum_samples(tmp_path, monkeypatch):
    plan, _events = make_plan(tmp_path)
    payload = json.loads(plan.read_text())
    payload["soak"]["minimum_samples"] = 4
    plan.write_text(json.dumps(payload))
    monkeypatch.setattr(canary, "_MAX_SOAK_SAMPLES", 3)
    with pytest.raises(ValueError, match="minimum_samples.*limit"):
        canary.load_plan(plan)
def test_soak_evidence_budget_exhaustion_triggers_verified_rollback(tmp_path, monkeypatch):
    plan, events = make_plan(tmp_path)
    monkeypatch.setattr(canary, "_MAX_SOAK_BYTES", 1)
    workspace = tmp_path / "runs"
    with pytest.raises(RuntimeError, match="soak evidence size limit.*rollback_succeeded=True"):
        canary.execute(plan, workspace, "soak-budget-rollback", False)
    assert events.read_text().splitlines()[-3:] == [
        "stop", "rollback", "rollback-health",
    ]
    report = canary.verify_manifest(workspace / "soak-budget-rollback")
    assert report["success"] is False
    assert report["rollback_succeeded"] is True
    assert report["admission"] == {"admitted": False}
    with pytest.raises(ValueError, match="did not complete successfully"):
        canary.verify_manifest(workspace / "soak-budget-rollback", require_success=True)
@pytestmark_attestation
def test_rehashed_attestation_cannot_misattribute_signature_to_another_allowed_key(tmp_path):
    plan, _events = make_plan(tmp_path)
    workspace = tmp_path / "runs"
    canary.execute(plan, workspace, "key-attribution", False)
    root = workspace / "key-attribution"
    primary = tmp_path / "primary-key"
    unrelated = tmp_path / "other-key"
    for key in (primary, unrelated):
        subprocess.run(
            ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)],
            check=True, capture_output=True,
        )
        key.chmod(0o600)
    allowed = tmp_path / "allowed_signers"
    allowed.write_text(
        "".join(
            "runtime-canary-operator " + key.with_suffix(".pub").read_text(encoding="utf-8")
            for key in (primary, unrelated)
        ), encoding="utf-8",
    )
    attestation.sign_run(root, primary, require_success=True)
    verified = attestation.verify_attestation(root, allowed, "runtime-canary-operator", require_success=True)
    primary_fingerprint = attestation._ssh._key_fingerprint(primary)
    other_fingerprint = attestation._ssh._key_fingerprint(unrelated)
    assert verified["signing_key_fingerprint"] == primary_fingerprint
    assert other_fingerprint != primary_fingerprint

    # Forge only unsigned JSON and its ordinary recomputable hash. SSHSIG
    # remains signed by the primary key, even though both keys are authorized.
    attestation_path = root / "runtime-canary-attestation.json"
    payload = json.loads(attestation_path.read_text(encoding="utf-8"))
    payload["signing_key_fingerprint"] = other_fingerprint
    core = {
        key: val for key, val in payload.items()
        if key not in {"created_at_utc", "attestation_fingerprint"}
    }
    payload["attestation_fingerprint"] = canary._operator.canonical_hash(core)
    attestation_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="signing key fingerprint"):
        attestation.verify_attestation(root, allowed, "runtime-canary-operator", require_success=True)
