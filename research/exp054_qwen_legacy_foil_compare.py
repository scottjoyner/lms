#!/usr/bin/env python3
"""Machine-only source-specificity comparison: original vs new Qwen conditions.

Both cohorts share same 64 article and frozen original wrong-exam answer, but
the newer run uses truncated, token-gap-bounded passage prefixes, 5.20.1
Transformers vs old 5.12.1, and all three foils. Differences are confounded:
NOT a causal estimate of semantic evidence truncation.
"""
import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path

import exp053_agent_qwen_cpu as old
import exp054_qwen_all_foils_cpu as new
import exp054_qwen_threefoil_audit as auditor
from exp053_tokenizer_preflight import validate_custody

SCHEMA="exp054-qwen-frozen-original-foil-paired-descriptive-v1"
BOOTSTRAP=2000
SEED=54054


def hold(reason):
    raise ValueError("HOLD EXP054 PAIRED ORIGINAL FOIL: "+reason)


def per_article(original,latest,foil):
    old_rows={(r["arm"],r["rotation"]):r for r in original["conditions"]}
    new_rows={(r["arm"],r["rotation"]):r for r in latest["conditions"]}
    old_s=[];new_s=[]
    if len(old_rows)!=6 or len(new_rows)!=4:
        hold("missing original or new paired conditions")
    original_foil=foil.get("exp053_original_machine_foil")
    gold=foil.get("gold_label_private")
    if original_foil not in "ABCD" or gold not in "ABCD" or gold==original_foil:
        hold("invalid precommitted answer key or foil")
    for rotation in (0,1):
        left=old_rows.get(("supported",rotation))
        right=old_rows.get(("decoy",rotation))
        x=new_rows.get(("supported_prefix",rotation))
        y=new_rows.get(("decoy_prefix",rotation))
        if any(v is None for v in (left,right,x,y)):
            hold("source/rotation condition missing")
        good=new.rotate(gold,rotation)
        bad=new.rotate(original_foil,rotation)
        if (left["gold_label"]!=good or left["foil_label"]!=bad or
            right["gold_label"]!=good or right["foil_label"]!=bad or
            x["gold_label"]!=good or bad not in x["foil_labels"] or
            y["gold_label"]!=good or bad not in y["foil_labels"]):
            hold("old and new wrong exam-answer mappings differ")
        old_margin=(left["gold_logp"]-left["foil_logp"]) - (right["gold_logp"]-right["foil_logp"])
        new_margin=(x["logprob_by_label"][good]-x["logprob_by_label"][bad]) - (
                    y["logprob_by_label"][good]-y["logprob_by_label"][bad])
        if not all(math.isfinite(v) for v in (old_margin,new_margin)):
            hold("nonfinite source contrast")
        old_s.append(old_margin)
        new_s.append(new_margin)
    return {"old":statistics.mean(old_s),"new":statistics.mean(new_s),
            "old_rotations":old_s,"new_rotations":new_s}


def interval(diff):
    n=len(diff)
    if n!=64:
        hold("complete 64 article paired cohort required")
    rng=random.Random(SEED)
    draw=sorted(statistics.mean(diff[rng.randrange(n)] for _ in range(n)) for _ in range(BOOTSTRAP))
    return [round(draw[int(.025*BOOTSTRAP)],6),
            round(draw[int(.975*BOOTSTRAP)],6)]


def describe(data):
    if len(data)!=64:
        hold("wrong case count")
    before=[x["old"] for x in data]
    after=[x["new"] for x in data]
    diffs=[b-a for a,b in zip(before,after)]
    if not all(math.isfinite(v) for v in before+after+diffs):
        hold("nonfinite article comparison")
    return {
      "cases":64,
      "old_original_full_passage_S_mean":round(statistics.mean(before),6),
      "new_wordprefix_S_mean_for_same_original_foil":round(statistics.mean(after),6),
      "new_minus_old_mean":round(statistics.mean(diffs),6),
      "new_minus_old_median":round(statistics.median(diffs),6),
      "paired_article_bootstrap95_descriptive":interval(diffs),
      "old_positive_cases":sum(x>0 for x in before),
      "new_positive_cases":sum(x>0 for x in after),
      "flipped_article_mean_sign":sum((x>0)!=(y>0) for x,y in zip(before,after)),
      "new_all_rotations_positive_original_foil":sum(all(t>0 for t in x["new_rotations"]) for x in data),
      "human_adjudicated":False,
      "causal_prefix_effect_proven":False
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--write-private",action="store_true")
    args=p.parse_args()
    lookup,manifest,sources,foils,pinned=new.load_frozen_inputs()
    frozen=json.loads(old.FROZEN.read_bytes())["models"][old.NAME]
    old_code=old.sha(Path(old.__file__).read_bytes())
    new_code=new.sha(Path(new.__file__).read_bytes())
    source_sha=old.sha((new.PRECOMMIT/"exp054-wordprefix-balanced-index-v1.json").read_bytes())
    foil_sha=old.sha((new.PRECOMMIT/"exp054-all-three-machine-foils-v1.json").read_bytes())
    result=[];receipt_hashes=[]
    for i,cid in enumerate(sorted(lookup)):
        p_old=old.RECEIPTS/f"case-{i:03}.json"
        p_new=new.RECEIPTS/f"case-{i:03}.json"
        r_old=old.read_receipt(p_old)
        r_new=old.read_receipt(p_new)
        old.verify_case(r_old,i,cid,frozen["checkpoint_receipt_sha256"],
                        manifest["packet_sha256"],old_code)
        auditor.audit_case(r_new,i,cid,sources[i],foils["cases"][i],
            manifest["packet_sha256"],frozen["checkpoint_receipt_sha256"],
            new_code,source_sha,foil_sha,lookup)
        result.append(per_article(r_old,r_new,foils["cases"][i]))
        receipt_hashes.append({"cohort_index":i,
             "original_sha256":old.sha(p_old.read_bytes()),
             "new_sha256":old.sha(p_new.read_bytes())})
    if list(new.RECEIPTS.glob("failure-*.json")):
        hold("unreconciled new-case failures")
    summary=describe(result)
    record={
       "schema":SCHEMA,"human_adjudicated":False,"case_count":64,
       "original_forward_passes_reused":384,"new_forward_passes_reused":256,
       "extra_model_forwards":0,"original_study_completed":True,
       "source_packet_sha256":manifest["packet_sha256"],
       "original_runner_sha256":old_code,"new_runner_sha256":new_code,
       "source_control_index_sha256":source_sha,
       "all_three_foils_sha256":foil_sha,
       "paired_case_receipts":receipt_hashes,
       "descriptive":summary,
       "limitations":"Original versus new source wording, truncation, control construction, runtime versions and all-foil scoring differ. This is retrospective paired engineering evidence, NOT a causal test or scientific confirmation. Full passage evidence sufficiency and model training-contamination remain unverified."
    }
    if args.write_private:
        old.private_dir(new.RECEIPTS)
        name="exp054-qwen-legacy-original-foil-paired-v1.json"
        dst=new.RECEIPTS/name
        expected=(json.dumps(record,sort_keys=True,separators=(",",":"))+"\n").encode()
        if dst.exists() or dst.is_symlink():
            if old.read_receipt(dst)!=record or dst.read_bytes()!=expected:
                hold("prior frozen paired aggregate does not match")
            checksum=old.sha(expected)
            print("EXP054_LEGACY_PAIR_REUSED sha256="+checksum,flush=True)
        else:
            checksum=old.create_immutable(new.RECEIPTS,name,record)
            print("EXP054_LEGACY_PAIR_FROZEN sha256="+checksum,flush=True)
    print("EXP054_QWEN_LEGACY_ORIGINAL_FOIL_COMPARE",summary,flush=True)


if __name__=="__main__":
    main()
