"""Pure synthetic aggregate tests: never use private passage text or model inference."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))
import exp053_agent_k2_summary as agg


def row(supported=-1., withheld=-2., decoy=-1.5):
    rows=[]
    for rotation in (0,1):
        for arm,score in [("supported",supported),("withheld",withheld),("decoy",decoy)]:
            rows.append({"arm":arm,"rotation":rotation,"gold_logp":float(score),
                         "foil_logp":-4.0, "prompt_sha256":"a"*64})
    return {"conditions":rows}


def test_synthetic_descriptive_metrics():
    vals=agg.summarize([row(),row(supported=-3,withheld=-2,decoy=-2.5)],count=2)
    assert vals["D"]["mean"]==0.0 and vals["D"]["positive"]==1
    assert vals["S"]["mean"]==0.0 and vals["S"]["positive"]==1
    assert vals["N"]["positive"]==1
    assert vals["D"]["n"]==2


@pytest.mark.parametrize("mutation",[
    lambda rec:rec["conditions"].pop(),
    lambda rec:rec["conditions"].append(rec["conditions"][0]),
    lambda rec:rec["conditions"][0].update(gold_logp=float("nan")),
    lambda rec:rec["conditions"][0].update(gold_logp=0.3),
    lambda rec:rec["conditions"][0].update(rotation=3),
    lambda rec:rec["conditions"][0].update(gold_logp="not-a-number"),
])
def test_bad_synthetic_scores_fail_closed(mutation):
    x=row()
    mutation(x)
    with pytest.raises(ValueError,match="HOLD"):
        agg.case_margins(x)


def test_missing_synthetic_cases_fail_closed():
    with pytest.raises(ValueError,match="HOLD"):
        agg.summarize([row()],count=2)
    with pytest.raises(ValueError,match="HOLD"):
        agg.summarize([],count=0)


def test_resume_verifies_same_immutable_json(tmp_path):
    import os
    os.chmod(tmp_path, 0o700)
    expected={"schema":"synthetic-resume","case_count":2,
              "human_adjudicated":False,"receipts":["opaque"]}
    digest=agg.create_immutable(tmp_path,"summary.json",expected)
    assert agg.verify_existing_aggregate(tmp_path/"summary.json",expected)==digest
    with pytest.raises(ValueError,match="HOLD"):
        agg.verify_existing_aggregate(tmp_path/"summary.json",{**expected,"case_count":3})


def test_resume_rejects_ambiguous_json(tmp_path):
    import os
    os.chmod(tmp_path,0o700)
    p=tmp_path/"summary.json"
    p.write_text('{"case_count":2,"case_count":2}\n')
    os.chmod(p,0o600)
    with pytest.raises(ValueError,match="HOLD"):
        agg.verify_existing_aggregate(p,{"case_count":2})


def test_resume_rejects_symlink(tmp_path):
    import os
    os.chmod(tmp_path,0o700)
    a=tmp_path/"real.json"
    a.write_text('{"case_count":2}\n')
    os.chmod(a,0o600)
    link=tmp_path/"linked.json"
    link.symlink_to(a)
    with pytest.raises(RuntimeError,match="HOLD"):
        agg.verify_existing_aggregate(link,{"case_count":2})
