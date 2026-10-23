"""Synthetic receipt tamper and source-identity tests for EXP-054 all-foil auditor."""
import copy
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"research"))
import exp054_qwen_threefoil_audit as audit


def fixture():
    cid="case-synthetic-opaque"
    lookup={cid:{"question":"Which answer?","article":"The synthetic supporting article has paragraphs. "*30,
                 "options":{"A":"choice a","B":"choice b","C":"choice c","D":"choice d"}},
            "decoy-synthetic":{"article":"Entirely unrelated synthetic article. "*50}}
    foils={"case_id":cid,"gold_label_private":"A",
           "all_three_machine_wrong_labels":["B","C","D"],
           "exp053_original_machine_foil":"B"}
    rows=[]; frozen=[]
    for rot in (0,1):
        for arm in audit.score.ARMS:
            passage=lookup[cid if arm=="supported_prefix" else "decoy-synthetic"]["article"][:300]
            s=audit.prompt(lookup[cid]["question"],audit.options_for(lookup[cid],rot),passage)
            psha=audit.digest(s)
            gold=audit.score.rotate("A",rot)
            wrong=[audit.score.rotate(x,rot) for x in ("B","C","D")]
            scores={letter:-3. for letter in "ABCD"}
            scores[gold]=-0.5 if arm=="supported_prefix" else -1.5
            rows.append({"rotation":rot,"arm":arm,"prompt_sha256":psha,
                         "input_tokens":100,"gold_label":gold,
                         "foil_labels":wrong,"logprob_by_label":scores})
            frozen.append({"rotation":rot,"arm":arm,"prompt_sha256":psha,
                           "input_tokens":{audit.score.NAME:100}})
    source={"case_id":cid,"cohort_index":0,"decoy_case_id":"decoy-synthetic",
            "supported_prefix_codepoints":300,"decoy_prefix_codepoints":300,
            "conditions":frozen}
    receipt={
      "schema":audit.score.SCHEMA,"cohort_index":0,"case_id":cid,
      "private_packet_sha256":"a"*64,"checkpoint_receipt_sha256":"b"*64,
      "runner_sha256":"c"*64,"source_control_index_sha256":"d"*64,
      "frozen_three_foils_sha256":"e"*64,
      "human_adjudicated":False,"machine_only":True,
      "semantic_evidence_preserved_verified":False,
      "model_forward_passes":4,"conditions":rows
    }
    return receipt,source,foils,lookup


def verify(p,s,f,l):
    return audit.audit_case(p,0,"case-synthetic-opaque",s,f,
                            "a"*64,"b"*64,"c"*64,"d"*64,"e"*64,l)


def test_all_three_foils_scored_each_rotation():
    p,s,f,l=fixture()
    out=verify(p,s,f,l)
    assert set(out)==set("BCD")
    assert all(v==[1.,1.] for v in out.values())
    summ=audit.summarize([out,out])
    assert summ["article_count"]==2
    assert summ["number_of_foil_comparisons"]==6
    assert summ["all_foils_both_rotations_positive_per_case"]==2


@pytest.mark.parametrize("mutate",[
 lambda p,s,f,l:p["conditions"][0].update(prompt_sha256="0"*64),
 lambda p,s,f,l:p["conditions"][0].update(input_tokens=99),
 lambda p,s,f,l:p["conditions"][0].update(gold_label="B"),
 lambda p,s,f,l:p["conditions"][0].update(foil_labels=["B","C","A"]),
 lambda p,s,f,l:p["conditions"][0]["logprob_by_label"].update(A=float("nan")),
 lambda p,s,f,l:p["conditions"][0]["logprob_by_label"].update(A=0.1),
 lambda p,s,f,l:s["conditions"][0].update(prompt_sha256="f"*64),
 lambda p,s,f,l:p.update(semantic_evidence_preserved_verified=True),
 lambda p,s,f,l:p.update(human_adjudicated=True),
 lambda p,s,f,l:f.update(gold_label_private="D"),
])
def test_bad_private_receipts_hold(mutate):
    args=list(fixture())
    mutate(*args)
    with pytest.raises((ValueError,RuntimeError),match="HOLD"):
        verify(*args)


def test_no_cohort_is_not_a_result():
    with pytest.raises(ValueError,match="HOLD"):
        audit.summarize([])


def test_all_three_required():
    with pytest.raises(ValueError,match="HOLD"):
        audit.summarize([{"B":[1.,2.]}])
