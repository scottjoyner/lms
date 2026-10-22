"""Synthetic-only negative tests for the autonomous *machine-only* scorer."""
import json
import os
from pathlib import Path
import stat
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))
import exp053_agent_qwen_cpu as pilot


def synthetic_receipt():
    rows = [{"arm": arm, "rotation": rotation,
             "prompt_sha256": "%064x" % (1 + rotation * 3 + i),
             "gold_label":"A", "foil_label":"B",
             "gold_logp":-0.5, "foil_logp":-1.5}
            for rotation in (0, 1) for i, arm in enumerate(pilot.ARM_ORDER)]
    return {"schema":pilot.SCHEMA, "cohort_index":0, "case_id":"opaque",
            "checkpoint_receipt_sha256":"checkpoint", "private_packet_sha256":"packet",
            "runner_sha256":"runner", "machine_only":True,
            "human_adjudicated":False, "conditions":rows}


def verify(x):
    pilot.verify_case(x, 0, "opaque", "checkpoint", "packet", "runner")


def test_complete_synthetic_receipt():
    verify(synthetic_receipt())


@pytest.mark.parametrize("mutator", [
    lambda r: r.update(machine_only=False),
    lambda r: r.update(human_adjudicated=True),
    lambda r: r.update(runner_sha256="swapped"),
    lambda r: r.update(private_packet_sha256="changed"),
    lambda r: r.update(checkpoint_receipt_sha256="changed"),
    lambda r: r["conditions"].pop(),
    lambda r: r["conditions"][1].update(prompt_sha256=r["conditions"][0]["prompt_sha256"]),
    lambda r: r["conditions"][0].update(foil_label="A"),
    lambda r: r["conditions"][0].update(gold_logp=float("nan")),
    lambda r: r["conditions"][0].update(rotation=2),
])
def test_reject_tampered_or_incomplete_receipt(mutator):
    r = synthetic_receipt()
    mutator(r)
    with pytest.raises(RuntimeError, match="HOLD"):
        verify(r)


def test_atomic_owner_private_receipt_cannot_be_overwritten(tmp_path):
    os.chmod(tmp_path,0o700)
    pilot.private_dir(tmp_path)
    data=synthetic_receipt()
    checksum=pilot.save_case(tmp_path,0,data)
    path=tmp_path/"case-000.json"
    raw=path.read_bytes()
    assert pilot.sha(raw)==checksum
    assert stat.S_IMODE(path.stat().st_mode)==0o600
    assert json.loads(raw)==data
    assert not list(tmp_path.glob(".pending-*"))
    with pytest.raises(FileExistsError):
        pilot.save_case(tmp_path,0,{"malicious":"overwrite"})
    assert path.read_bytes()==raw
    assert not list(tmp_path.glob(".pending-*"))


def test_reject_symlink_or_world_visible_root(tmp_path):
    other=tmp_path/"real"
    other.mkdir(mode=0o700)
    link=tmp_path/"link"
    link.symlink_to(other,target_is_directory=True)
    with pytest.raises(RuntimeError,match="symlinked"):
        pilot.private_dir(link)
    os.chmod(other,0o755)
    with pytest.raises(RuntimeError,match="owner-only"):
        pilot.private_dir(other)


def test_failure_journal_redacts_exception_message(tmp_path):
    os.chmod(tmp_path,0o700)
    pilot.journal_failure(tmp_path,3,RuntimeError("SECRET_SENTINEL_prompt_data"))
    names=list(tmp_path.iterdir())
    assert len(names)==1
    obj=json.loads(names[0].read_bytes())
    assert obj["status"]=="INCOMPLETE_DO_NOT_USE_AS_SCORE"
    assert obj["failure_type"]=="RuntimeError"
    assert "SECRET_SENTINEL" not in names[0].read_text()
    assert stat.S_IMODE(names[0].stat().st_mode)==0o600