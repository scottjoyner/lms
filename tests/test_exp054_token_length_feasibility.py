"""Synthetic tokenizer-feasibility tests; no model checkpoints or RACE corpus."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"research"))
import exp054_token_length_feasibility as module
import exp054_prefix_length_control as pair


class FakeTokenizer:
    def encode(self,text,add_special_tokens=False):
        # Simple stable whitespace tokenizer with four independent label IDs.
        return [sum(map(ord,w)) for w in text.split()]


class MultiTokenLabels(FakeTokenizer):
    def encode(self,text,add_special_tokens=False):
        tokens=super().encode(text,add_special_tokens)
        if text.strip() in "ABCD" and len(text.strip())==1:
            return tokens+[9]
        return tokens


def synthetic():
    letters="abcdefghijklmnopqrstuvwxyz"
    rows={f"case_{i:03}":{
        "article":("source%03d "%i)+(letters if i%2 else letters[::-1])*100,
        "question":"Why or how?",
        "options":{"A":"alpha","B":"beta","C":"gamma","D":"delta"}}
        for i in range(64)}
    admit={"human_approval":False,"accepted":{cid:{} for cid in rows}}
    manifest=pair.build_manifest(rows,admit,"a"*64)
    return rows,manifest


def test_all_model_prompt_boundaries_synthetic():
    rows,manifest=synthetic()
    prompts=module.enumerate_prompts(rows,manifest)
    assert len(prompts)==64
    stats=module.token_counts(FakeTokenizer(),prompts)
    assert stats["total_prompt_count"]==256
    assert stats["label_boundary_checks"]==1024
    assert stats["condition_count"]==128
    assert len(stats["rows"])==64
    assert stats["exact_token_matches"]<=128
    assert stats["absolute_gap_max"]>=0


def test_model_label_multitoken_hold():
    rows,manifest=synthetic()
    with pytest.raises(ValueError,match="HOLD"):
        module.token_counts(MultiTokenLabels(),module.enumerate_prompts(rows,manifest))


def test_source_prompt_hash_tamper_holds():
    rows,manifest=synthetic()
    manifest["cases"][0]["controls"][0]["prompt_sha256"]="0"*64
    with pytest.raises(ValueError,match="HOLD"):
        module.enumerate_prompts(rows,manifest)


def test_unrelated_corrupt_decoy_fails():
    rows,manifest=synthetic()
    rows["case_001"]["article"]="short"
    with pytest.raises(ValueError,match="HOLD"):
        module.enumerate_prompts(rows,manifest)
