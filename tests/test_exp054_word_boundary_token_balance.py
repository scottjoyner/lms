"""Synthetic-only prospective tokenizer-match optimizer tests."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
import exp054_word_boundary_token_balance as balance


def test_deterministic_selection_prefers_long_source():
    a=[(300,(70,70,66)),(800,(210,210,200)),(1200,(320,320,312))]
    b=[(320,(72,72,68)),(780,(211,211,201)),(1190,(322,322,314))]
    assert balance.choose(a,b)==(800,780,(210,210,200),(211,211,201))


def test_gap_out_of_bounds_does_not_select():
    with pytest.raises(ValueError,match="HOLD"):
        balance.choose([(900,(200,200,200))],[(900,(230,230,230))])


def test_char_ratio_fails_closed():
    with pytest.raises(ValueError,match="HOLD"):
        balance.choose([(1600,(200,200,200))],[(500,(200,200,200))])


def test_exact_model_token_counts_accepted():
    assert balance.choose([(1000,(231,231,220))],[(900,(231,231,220))])[:2]==(1000,900)


def test_tie_determinism():
    a=[(1024,(200,200,190)),(1020,(202,202,192))]
    b=[(1000,(202,202,192)),(1003,(201,201,191))]
    assert balance.choose(a,b)==balance.choose(a,b)


def test_source_fixture_word_boundary_no_midword():
    text=("alpha beta gamma delta "*120)
    positions=balance.word_boundary_prefixes(text)
    assert positions and max(positions)<=1600 and min(positions)>=256
    assert all(text[p-1].isspace() for p in positions)


def test_no_model_logits_or_scoring_in_analysis():
    s=(Path(__file__).resolve().parents[1]/"research"/"exp054_word_boundary_token_balance.py").read_text()
    assert "AutoModel" not in s and "from_pretrained" in s
    assert "logits[" not in s and "torch." not in s


def test_manifest_freeze_requires_64_cases():
    with pytest.raises(ValueError,match="HOLD"):
        balance.root_manifest([], "a"*64, "b"*64)


def test_immutable_partitioned_private_report(tmp_path,monkeypatch):
    import os
    import json
    import hashlib
    os.chmod(tmp_path,0o700)
    monkeypatch.setattr(balance,"RECEIPTS",tmp_path)
    cases=[{"case_id":f"synthetic-{i:03d}",
            "controls":[{"rotation":0,"arm":"supported_prefix"}]}
           for i in range(64)]
    report=balance.root_manifest(cases,"a"*64,"b"*64)
    digest=balance.publish_report(report)
    assert len(list(tmp_path.glob("exp054-wordprefix-balanced-part*.json")))==8
    parent=tmp_path/"exp054-wordprefix-balanced-index-v1.json"
    assert digest==hashlib.sha256(parent.read_bytes()).hexdigest()
    assert json.loads(parent.read_bytes())["case_count"]==64
    assert balance.publish_report(report)==digest
    assert all((p.stat().st_mode & 0o777)==0o600 for p in tmp_path.iterdir())


def test_immutable_partition_tamper_holds(tmp_path,monkeypatch):
    import os
    os.chmod(tmp_path,0o700)
    monkeypatch.setattr(balance,"RECEIPTS",tmp_path)
    report=balance.root_manifest([{"case_id":f"synthetic-{i:03d}"} for i in range(64)],
                                  "a"*64,"b"*64)
    balance.publish_report(report)
    path=tmp_path/"exp054-wordprefix-balanced-part00-v1.json"
    path.write_text('{"tampered":true}\n')
    with pytest.raises((ValueError,RuntimeError),match="HOLD"):
        balance.publish_report(report)
