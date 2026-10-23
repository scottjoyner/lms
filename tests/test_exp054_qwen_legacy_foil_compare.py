"""Synthetic-only paired old/new Qwen foil source contrast tests."""
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"research"))
import exp054_qwen_legacy_foil_compare as module


def fixture():
    foil={"gold_label_private":"A","exp053_original_machine_foil":"B"}
    old=[];new=[]
    for r in (0,1):
        good=module.new.rotate("A",r);bad=module.new.rotate("B",r)
        for arm in ("supported","withheld","decoy"):
            old.append({"arm":arm,"rotation":r,"gold_label":good,"foil_label":bad,
                        "gold_logp":-1. if arm=="supported" else -2.,
                        "foil_logp":-4.})
        for arm in ("supported_prefix","decoy_prefix"):
            values={l:-3. for l in "ABCD"}
            values[good]=-1. if arm=="supported_prefix" else -1.5
            new.append({"arm":arm,"rotation":r,
                        "gold_label":good,
                        "foil_labels":[module.new.rotate(x,r) for x in "BCD"],
                        "logprob_by_label":values})
    return {"conditions":old},{"conditions":new},foil


def test_paired_original_foil_fixed():
    a,b,f=fixture()
    result=module.per_article(a,b,f)
    assert result["old"]==1.
    assert result["new"]==.5
    assert result["old_rotations"]==[1.,1.]
    assert result["new_rotations"]==[.5,.5]


def test_comparison_summary_and_seed():
    p=module.per_article(*fixture())
    x=module.describe([p]*64)
    assert x==module.describe([p]*64)
    assert x["old_positive_cases"]==64
    assert x["new_positive_cases"]==64
    assert x["new_minus_old_mean"]==-.5
    assert x["paired_article_bootstrap95_descriptive"]==[-.5,-.5]
    assert x["causal_prefix_effect_proven"] is False


@pytest.mark.parametrize("mutate",[
 lambda a,b,f:f.update(exp053_original_machine_foil="A"),
 lambda a,b,f:a["conditions"][0].update(gold_label="C"),
 lambda a,b,f:b["conditions"][0].update(foil_labels=["C","D"]),
 lambda a,b,f:a["conditions"].pop(),
 lambda a,b,f:b["conditions"].pop(),
 lambda a,b,f:b["conditions"][0]["logprob_by_label"].update(A=float("nan")),
])
def test_incorrect_condition_or_label_holds(mutate):
    args=list(fixture())
    mutate(*args)
    with pytest.raises((ValueError,KeyError,StopIteration),match="HOLD|nan"):
        module.per_article(*args)


def test_wrong_cohort_size_refused():
    p=module.per_article(*fixture())
    with pytest.raises(ValueError,match="HOLD"):
        module.describe([p]*63)
