"""EXP-054 synthetic foil-proxy diagnostics; zero dataset, model weights or inference."""
import copy
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"research"))
import exp054_machine_proxy_robustness as diag


def case(s=2.,w=0.,d=-1.,tokens=(150,12,150)):
    conditions=[]
    for r in (0,1):
        for arm,margin,n in zip(diag.ARMS,(s,w,d),tokens):
            conditions.append({"arm":arm,"rotation":r,
                "gold_logp":float(margin-5),
                "foil_logp":-5.,
                "input_tokens":n,
                "gold_label":"A" if r==0 else "C",
                "foil_label":"B" if r==0 else "D",
                "prompt_sha256":f"{(len(conditions)+1):064x}"})
    return {"conditions":conditions}


def test_exact_descriptive_margins():
    p=diag.per_case(case())
    assert p["contrasts"]=={"D":2.,"N":-1.,"S":3.}
    assert p["tokens"]=={"D":138,"N":138,"S":0}
    assert p["token_gaps_by_rotation"]["S"] == [0,0]
    assert p["by_rotation"]["D"]==[2.,2.]


def test_spearman_ranks_and_ties():
    assert diag.spearman([1,2,3],[3,2,1])==-1.
    assert diag.spearman([1,2,3],[1,2,3])==1.
    assert diag.spearman([1,1,1],[1,2,3]) is None


def test_64_article_trim_and_leave_one_out():
    rows=[diag.per_case(case(s=2+i/32,w=0,d=0,tokens=(150+i,10,120+i)))
          for i in range(64)]
    output=diag.diagnose(rows,"Synthetic")
    assert output["D"]["n_articles"]==64
    assert output["D"]["positive"]==64
    assert output["D"]["both_rotations_positive"]==64
    assert output["D"]["rotation_sign_disagree"]==0
    assert output["D"]["token_gap_exact_matches"] == 0
    assert output["S"]["token_gap_exact_matches"] == 0
    assert output["D"]["all_leave_one_out_means_positive"] is True
    assert output==diag.diagnose(rows,"Synthetic")


def test_rotation_disagreement_detected():
    rows=[diag.per_case(case()) for _ in range(64)]
    x=copy.deepcopy(rows[0])
    x["by_rotation"]["D"]=[4.,-1.]
    x["contrasts"]["D"]=1.5
    rows[0]=x
    out=diag.diagnose(rows,"Synthetic")
    assert out["D"]["rotation_sign_disagree"]==1
    assert out["D"]["both_rotations_positive"]==63


@pytest.mark.parametrize("edit",[
    lambda c:c["conditions"].pop(),
    lambda c:c["conditions"].append(c["conditions"][0]),
    lambda c:c["conditions"][0].update(gold_logp=float("nan")),
    lambda c:c["conditions"][0].update(gold_logp=0.1),
    lambda c:c["conditions"][0].update(input_tokens=0),
    lambda c:c["conditions"][0].update(input_tokens=1.5),
    lambda c:c["conditions"][0].update(foil_label="A"),
    lambda c:c["conditions"][0].update(rotation=3),
])
def test_malformed_receipt_fails_closed(edit):
    c=case()
    edit(c)
    with pytest.raises(ValueError,match="HOLD"):
        diag.per_case(c)


def test_bad_counts_hold():
    with pytest.raises(ValueError,match="HOLD"):
        diag.diagnose([diag.per_case(case())]*63,"Synthetic")
    with pytest.raises(ValueError,match="HOLD"):
        diag.spearman([1,2],[1])
    with pytest.raises(ValueError,match="HOLD"):
        diag.spearman([1,2,3],[1,float("nan"),3])


def test_descriptive_bootstrap_seed_stable():
    x=[float(i) for i in range(64)]
    assert diag.bootstrap_bounds(x,"Synthetic:D")==diag.bootstrap_bounds(x,"Synthetic:D")
