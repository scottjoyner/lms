"""Synthetic verification of frozen alternative foil Qwen pilot; no model weights."""
import pytest
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
import exp054_qwen_all_foils_cpu as pilot


def source():
    return {"case_id":"fake","cohort_index":0,"decoy_case_id":"fake_next",
            "conditions":[{"rotation":r,"arm":a,"prompt_sha256":"a"*64}
                          for r in (0,1) for a in pilot.ARMS]}


def foil():
    return {"case_id":"fake","gold_label_private":"A",
            "all_three_machine_wrong_labels":["B","C","D"],
            "exp053_original_machine_foil":"D"}


def test_synthetic_source_and_labels():
    case={"case_id":"fake","article":"synthetic article"}
    rows=pilot.prepare_case(case,source(),foil(),None,None)
    assert len(rows)==4
    assert pilot.rotate("A",0)=="A"
    assert pilot.rotate("A",1)=="C"
    assert [pilot.rotate(c,1) for c in "BCD"]==["D","A","B"]


@pytest.mark.parametrize('bad',[
 {"gold_label_private":"A","all_three_machine_wrong_labels":["B","C","C"]},
 {"gold_label_private":"A","all_three_machine_wrong_labels":["B","C","X"]},
 {"gold_label_private":"A","all_three_machine_wrong_labels":["A","C","D"]},
])
def test_invalid_alternative_labels_holds(bad):
    f=foil()
    f.update(bad)
    with pytest.raises(ValueError,match="HOLD"):
        pilot.prepare_case({"case_id":"fake","article":"x"},source(),f,None,None)


def test_bad_condition_order_holds():
    s=source()
    s["conditions"].reverse()
    with pytest.raises(ValueError,match="HOLD"):
        pilot.prepare_case({"case_id":"fake","article":"x"},s,foil(),None,None)


def test_existing_receipt_mismatched_source_holds():
    receipt={"schema":pilot.SCHEMA,"cohort_index":0,"case_id":"fake",
            "private_packet_sha256":"a"*64,"checkpoint_receipt_sha256":"b"*64,
            "runner_sha256":"c"*64,"source_control_index_sha256":"d"*64,
            "frozen_three_foils_sha256":"e"*64,"human_adjudicated":False,
            "model_forward_passes":4,
            "conditions":[{"arm":arm,"rotation":rot,"gold_label":"A",
                           "foil_labels":["B","C","D"],
                           "logprob_by_label":{c:-2. for c in "ABCD"}}
                          for rot in (0,1) for arm in pilot.ARMS]}
    pilot.check_receipt(receipt,0,"fake","a"*64,"b"*64,"c"*64,"d"*64,"e"*64)
    with pytest.raises(ValueError,match="HOLD"):
        pilot.check_receipt(receipt,0,"fake","a"*64,"b"*64,"c"*64,"0"*64,"e"*64)
    receipt["conditions"][0]["logprob_by_label"]["A"]=float("nan")
    with pytest.raises(ValueError,match="HOLD"):
        pilot.check_receipt(receipt,0,"fake","a"*64,"b"*64,"c"*64,"d"*64,"e"*64)
