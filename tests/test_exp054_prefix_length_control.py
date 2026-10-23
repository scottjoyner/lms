"""Synthetic-only source-length prefix construction; no private RACE or model imports."""
import copy
import sys
from pathlib import Path
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"research"))
import exp054_prefix_length_control as ctl


def fixture():
    alphabet="abcdefghijklmnopqrstuvwxyz"
    lookup={}
    accepted={}
    for i in range(64):
        cid=f"synthetic_{i:03}"
        passage=("[SRC%02d] "%i) + (alphabet[::-1] if i%2 else alphabet)*80
        lookup[cid]={"case_id":cid,
                     "article":passage,
                     "question":"What follows from source?",
                     "options":{"A":"one","B":"two","C":"three","D":"four"}}
        accepted[cid]={"supported_exam_key":"A","other_exam_option_proxy":"B"}
    return lookup,{"accepted":accepted,"human_approval":False}


def test_deterministic_with_character_matched_prefixes():
    lookup,source=fixture()
    result=ctl.build_manifest(lookup,source,"a"*64)
    assert result==ctl.build_manifest(lookup,source,"a"*64)
    assert result["source_case_count"]==64
    assert result["prompts_per_model"]==256
    assert result["new_model_forward_passes"]==0
    assert result["token_length_parity_verified"] is False
    assert result["semantic_answer_evidence_retained_verified"] is False
    assert all(len(c["controls"])==4 for c in result["cases"])
    for row in result["cases"]:
        assert row["codepoint_budget"]==1600
        assert {r["passage_codepoints"] for r in row["controls"]}=={1600}
        assert len({r["prompt_sha256"] for r in row["controls"]})==4
    assert result["cases"][-1]["decoy_case_id"]=="synthetic_000"


def test_case_prefix_is_stable_and_not_duplicated():
    lookup,_=fixture()
    a=lookup["synthetic_001"]
    b=lookup["synthetic_002"]
    pair=ctl.make_pair(a,b)
    assert pair["supported_prefix"]==a["article"][:1600]
    assert pair["decoy_prefix"]==b["article"][:1600]
    assert len(pair["supported_prefix"])==len(pair["decoy_prefix"])


@pytest.mark.parametrize("mutate",[
 lambda lookup,adm:lookup.pop("synthetic_000"),
 lambda lookup,adm:adm["accepted"].pop("synthetic_000"),
 lambda lookup,adm:adm.update(human_approval=True),
 lambda lookup,adm:lookup["synthetic_000"].update(article="short"),
 lambda lookup,adm:lookup["synthetic_000"].update(article=lookup["synthetic_001"]["article"]),
 lambda lookup,adm:lookup["synthetic_000"].update(article=None),
 lambda lookup,adm:lookup["synthetic_000"].update(question=None),
 lambda lookup,adm:lookup["synthetic_000"].update(options={"A":"one"}),
])
def test_invalid_source_control_fails_closed(mutate):
    lookup,adm=fixture()
    mutate(lookup,adm)
    with pytest.raises((ValueError,TypeError,KeyError),match="HOLD|subscript"):
        ctl.plan(lookup,adm)


def test_invalid_budget_rejected():
    a={"article":"abcde"*300}
    b={"article":"defgh"*300}
    for n in (None,-1,100,1601,123.5):
        with pytest.raises(ValueError,match="HOLD"):
            ctl.make_pair(a,b,n)


def test_never_read_model_scores():
    s=(Path(__file__).resolve().parents[1]/"research"/"exp054_prefix_length_control.py").read_text()
    assert "from_pretrained" not in s
    assert "logits[" not in s
    assert "model(" not in s
