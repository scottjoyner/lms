"""Synthetic-only all-incorrect-exam-option precommit, independent of model logits."""
import copy
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"research"))
from exp054_all_wrong_option_preflight import build_manifest,LABELS


def fixture():
    return {"human_approval":False,"packet_sha256":"a"*64,
            "accepted":{f"synthetic_{i:03}":{
                "supported_exam_key":LABELS[i%4],
                "other_exam_option_proxy":LABELS[(i+1)%4]}
                for i in range(64)}}


def test_all_alternatives_are_frozen_without_models():
    result=build_manifest(fixture())
    assert result["exact_case_count"]==64
    assert result["new_model_forwards"]==0
    assert result["based_on_target_model_scores"] is False
    assert result["human_adjudicated"] is False
    assert result==build_manifest(fixture())
    for row in result["cases"]:
        assert row["gold_label_private"] not in row["all_three_machine_wrong_labels"]
        assert len(row["all_three_machine_wrong_labels"])==3
        assert set(row["all_three_machine_wrong_labels"]) | {row["gold_label_private"]}==set(LABELS)
        assert row["exp053_original_machine_foil"] in row["all_three_machine_wrong_labels"]


@pytest.mark.parametrize("mutate",[
 lambda f:f.update(human_approval=True),
 lambda f:f.update(packet_sha256="short"),
 lambda f:f["accepted"].pop("synthetic_000"),
 lambda f:f["accepted"]["synthetic_000"].update(supported_exam_key="X"),
 lambda f:f["accepted"]["synthetic_000"].update(other_exam_option_proxy="A"),
 lambda f:f["accepted"]["synthetic_000"].update(other_exam_option_proxy="X"),
 lambda f:f["accepted"]["synthetic_000"].clear(),
])
def test_invalid_custody_holds(mutate):
    f=fixture()
    mutate(f)
    with pytest.raises(ValueError,match="HOLD"):
        build_manifest(f)
