#!/usr/bin/env python3
"""Strict read-only EXP-054 Qwen original-triple-foil pilot verification.

Verifies raw private immutable case receipts and all frozen source/foil prompt
identities; publishes only a private aggregate. No model or tokenizer loads.
The machine-selected exam options are NOT human-reviewed plausible foils.
"""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import exp054_qwen_all_foils_cpu as score
from exp053_agent_qwen_cpu import FROZEN,create_immutable,private_dir,read_receipt,sha
from exp054_prefix_length_control import digest,prompt,options_for

AGGREGATE_SCHEMA="exp054-qwen-threefoil-pilot-aggregate-v1"


def hold(reason):
    raise ValueError("HOLD EXP054 THREE-FOIL AUDIT: "+reason)


def audit_case(row,idx,cid,source,foil,packet_sha,checkpoint_sha,runner_sha,source_sha,foil_sha,lookup):
    score.check_receipt(row,idx,cid,packet_sha,checkpoint_sha,runner_sha,source_sha,foil_sha)
    if row.get("machine_only") is not True or row.get("semantic_evidence_preserved_verified") is not False:
        hold("machine-only and semantic evidence claim limits")
    frozen_rows={(x["arm"],x["rotation"]):x for x in source["conditions"]}
    s_by_foil={l:[] for l in foil["all_three_machine_wrong_labels"]}
    for condition in row["conditions"]:
        arm,rotation=condition["arm"],condition["rotation"]
        frozen=frozen_rows.get((arm,rotation))
        if not frozen:
            hold("missing frozen source condition")
        current=lookup[cid]
        other=lookup[source["decoy_case_id"]]
        length=source["supported_prefix_codepoints" if arm=="supported_prefix" else "decoy_prefix_codepoints"]
        text=(current if arm=="supported_prefix" else other)["article"][:length]
        full_prompt=prompt(current["question"],options_for(current,rotation),text)
        if digest(full_prompt)!=frozen["prompt_sha256"] or condition.get("prompt_sha256")!=frozen["prompt_sha256"]:
            hold("prompt identity differs from frozen input")
        if condition.get("input_tokens")!=frozen["input_tokens"][score.NAME]:
            hold("input token count drift")
        gold=score.rotate(foil["gold_label_private"],rotation)
        wrong=[score.rotate(x,rotation) for x in foil["all_three_machine_wrong_labels"]]
        if condition.get("gold_label")!=gold or condition.get("foil_labels")!=wrong:
            hold("wrong answer mapping or rotation")
        values=condition["logprob_by_label"]
        if any(type(v)!=float or not math.isfinite(v) or v>0 for v in values.values()):
            hold("nonfinite or invalid next token logprobs")
        for f in foil["all_three_machine_wrong_labels"]:
            s_by_foil[f].append((arm,rotation,values[gold]-values[score.rotate(f,rotation)]))
    result={}
    for label,records in s_by_foil.items():
        if [(a,r) for a,r,_ in records]!=[(a,r) for r in (0,1) for a in score.ARMS]:
            hold("per-foil source arm/rotation structure incomplete")
        result[label]=[
            next(x for a,r,x in records if a=="supported_prefix" and r==rot) -
            next(x for a,r,x in records if a=="decoy_prefix" and r==rot)
            for rot in (0,1)
        ]
        if not all(math.isfinite(x) for x in result[label]):
            hold("nonfinite foil contrast")
    return result


def summarize(cases):
    if not cases:
        hold("no verified cases")
    # All three foils per case are a predeclared set, independent of scores.
    means=[];range_width=[]; all_foil_positive=0;all_rotation_positive=0
    all_foil_vals=[]
    for result in cases:
        if len(result)!=3:
            hold("three alternatives required")
        per_foil=[statistics.mean(v) for v in result.values()]
        means.append(statistics.mean(per_foil))
        all_foil_vals.extend(per_foil)
        range_width.append(max(per_foil)-min(per_foil))
        all_foil_positive+=int(all(v>0 for v in per_foil))
        all_rotation_positive+=int(all(v>0 for item in result.values() for v in item))
    return {
      "article_count":len(cases),
      "all_three_foil_average_S_mean":round(statistics.mean(means),6),
      "all_three_foil_S_average_positive":sum(x>0 for x in means),
      "all_three_foils_positive_per_case":all_foil_positive,
      "all_foils_both_rotations_positive_per_case":all_rotation_positive,
      "mean_within_case_foil_range_of_S":round(statistics.mean(range_width),6),
      "individual_foil_mean_S_positive":sum(x>0 for x in all_foil_vals),
      "number_of_foil_comparisons":len(all_foil_vals),
      "human_adjudicated":False,
      "semantic_evidence_retained_verified":False
    }


def run(count,write):
    if not 1<=count<=64:
        hold("invalid prefix count")
    lookup,manifest,source_cases,foils,pinned=score.load_frozen_inputs()
    checkpoint=json.loads(FROZEN.read_bytes())["models"][score.NAME]
    runner_sha=sha(Path(score.__file__).read_bytes())
    source_sha=sha((score.PRECOMMIT/"exp054-wordprefix-balanced-index-v1.json").read_bytes())
    foil_sha=sha((score.PRECOMMIT/"exp054-all-three-machine-foils-v1.json").read_bytes())
    results=[];hashes=[]
    for i in range(count):
        cid=sorted(lookup)[i]
        path=score.RECEIPTS/f"case-{i:03}.json"
        receipt=read_receipt(path)
        case=audit_case(receipt,i,cid,source_cases[i],foils["cases"][i],
                        manifest["packet_sha256"],checkpoint["checkpoint_receipt_sha256"],
                        runner_sha,source_sha,foil_sha,lookup)
        results.append(case)
        hashes.append({"cohort_index":i,"case_receipt_sha256":sha(path.read_bytes())})
    if list(score.RECEIPTS.glob("failure-*.json")):
        hold("unreconciled inference failure records")
    summary=summarize(results)
    aggregate={"schema":AGGREGATE_SCHEMA,"complete_frozen_cohort":count==64,
         "human_adjudicated":False,"source_evidence_retained_verified":False,
         "model":score.NAME,"case_count":count,"forward_pass_count":count*4,
         "all_wrong_exam_options_per_condition":3,
         "source_control_index_sha256":source_sha,
         "three_foil_roster_sha256":foil_sha,"runner_sha256":runner_sha,
         "model_checkpoint_receipt_sha256":checkpoint["checkpoint_receipt_sha256"],
         "case_receipts":hashes,"descriptive_pilot_statistics":summary,
         "claim_limit":"Exploratory machine-only, truncated prefix evidence sufficiency UNKNOWN; single model, selected cohort prefix and wrong exam options have no independent semantic adjudication."}
    if write:
        private_dir(score.RECEIPTS)
        name=f"summary-000-{count-1:03}-allfoil-v1.json"
        destination=score.RECEIPTS/name
        # Exact content revalidation, not replacement, if invoked a second time.
        canonical=(json.dumps(aggregate,sort_keys=True,separators=(",",":"))+"\n").encode()
        if destination.exists() or destination.is_symlink():
            if read_receipt(destination)!=aggregate or destination.read_bytes()!=canonical:
                hold("previous private aggregate bytes or provenance differ")
            h=hashlib.sha256(canonical).hexdigest()
            print("EXP054_QWEN_ALLFOIL_AGGREGATE_REUSED sha256="+h,flush=True)
        else:
            h=create_immutable(score.RECEIPTS,name,aggregate)
            print("EXP054_QWEN_ALLFOIL_AGGREGATE_FROZEN sha256="+h,flush=True)
    print("EXP054_QWEN_ALLFOIL_READONLY_AUDIT_OK articles="+str(count)+" forwards="+str(count*4)+" no_new_model_load=1",flush=True)
    print("EXP054_QWEN_PILOT_DESCRIPTIVE",summary,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count",type=int,default=8)
    parser.add_argument("--write-private",action="store_true")
    args=parser.parse_args()
    run(args.count,args.write_private)


if __name__=="__main__":
    main()
