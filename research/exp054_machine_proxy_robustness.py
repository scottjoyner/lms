#!/usr/bin/env python3
"""EXP-054: exploratory sensitivity of frozen EXP-053 machine-only case receipts.

Reads all 192 owner-private source-backed case receipts. No model imports, new
neural forwards, dataset text, human adjudications, or production mutations.
Writes only aggregate diagnostics with an exclusive owner-private write.
"""
import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path

from exp053_agent_three_model_compare import LABELS
from exp053_agent_qwen_cpu import (
    FROZEN, RECEIPTS as QROOT, private_dir, read_receipt,
    verify_case as verify_q, sha, create_immutable,
)
from exp053_agent_minicpm_cpu import RECEIPTS as MROOT, verify_case as verify_m
from exp053_agent_k2_cpu import RECEIPTS as KROOT, verify_case as verify_k
from exp053_agent_qwen_summary import source_data
from exp053_agent_pair_compare import case_compare

METRICS = ("D", "N", "S")
ARMS = ("supported", "withheld", "decoy")
ROTATIONS = (0, 1)
BOOTSTRAPS = 2000
SCHEMA = "exp054-existing-machine-proxy-robustness-v1"


def hold(message):
    raise ValueError("HOLD EXP054 ROBUSTNESS: " + message)


def get_grid(receipt):
    rows = receipt.get("conditions")
    if not isinstance(rows, list) or len(rows) != 6:
        hold("six source-arm/rotation conditions are required")
    out = {}
    for row in rows:
        key = (row.get("arm"), row.get("rotation"))
        if key not in {(a,r) for a in ARMS for r in ROTATIONS} or key in out:
            hold("invalid or duplicate passage-arm/rotation")
        for f in ("gold_logp", "foil_logp"):
            v = row.get(f)
            if type(v) is not float or not math.isfinite(v) or v > 0:
                hold("invalid source log probability")
        count = row.get("input_tokens")
        if type(count) is not int or count <= 0:
            hold("invalid token length")
        if row.get("gold_label") == row.get("foil_label"):
            hold("identical correct and foil labels")
        out[key] = row
    if len(out) != 6:
        hold("incomplete source-arm/rotation grid")
    return out


def per_case(receipt):
    grid = get_grid(receipt)
    differences = {m: [] for m in METRICS}
    lengths = {m:[] for m in METRICS}
    for rotation in ROTATIONS:
        margin = {arm:(grid[arm,rotation]["gold_logp"] -
                       grid[arm,rotation]["foil_logp"])
                  for arm in ARMS}
        t = {arm:grid[arm,rotation]["input_tokens"] for arm in ARMS}
        for metric, x, y in (("D","supported","withheld"),
                             ("N","decoy","withheld"),
                             ("S","supported","decoy")):
            differences[metric].append(margin[x] - margin[y])
            lengths[metric].append(t[x] - t[y])
    return {"contrasts":{k:statistics.mean(v) for k,v in differences.items()},
            "by_rotation":differences,
            "tokens":{k:statistics.mean(v) for k,v in lengths.items()},
            "token_gaps_by_rotation":lengths,
            "mean_input_tokens":{a:statistics.mean(
                grid[a,rotation]["input_tokens"] for rotation in ROTATIONS)
                for a in ARMS}}


def _ranks(values):
    indices = sorted(range(len(values)), key=lambda i:values[i])
    ranks=[0.]*len(values)
    i=0
    while i<len(indices):
        j=i+1
        while j<len(indices) and values[indices[j]]==values[indices[i]]:
            j+=1
        avg=(i+j-1)/2+1
        for pos in indices[i:j]:
            ranks[pos]=avg
        i=j
    return ranks


def spearman(x,y):
    if len(x)!=len(y) or len(x)<3:
        hold("invalid Spearman arrays")
    if not all(math.isfinite(v) for v in x+y):
        hold("nonfinite rank input")
    a,b=_ranks(x),_ranks(y)
    ma,mb=statistics.mean(a),statistics.mean(b)
    numerator=sum((v-ma)*(w-mb) for v,w in zip(a,b))
    da=sum((v-ma)**2 for v in a)
    db=sum((v-mb)**2 for v in b)
    if da==0 or db==0:
        return None
    return round(numerator/math.sqrt(da*db),6)


def bootstrap_bounds(values, tag):
    n=len(values)
    if n<2:hold("insufficient articles for bootstrap")
    h=hashlib.sha256(("EXP054-article-bootstrap-v1:"+tag).encode()).digest()
    rng=random.Random(int.from_bytes(h[:8],"big"))
    draws=sorted(sum(values[rng.randrange(n)] for _ in range(n))/n
                 for _ in range(BOOTSTRAPS))
    return [round(draws[int(BOOTSTRAPS*.025)],6),
            round(draws[int(BOOTSTRAPS*.975)],6)]


def diagnose(case_outputs, model_name):
    if len(case_outputs)!=64:
        hold("exactly 64 frozen articles required")
    result={}
    for metric in METRICS:
        vals=[row["contrasts"][metric] for row in case_outputs]
        left=[row["by_rotation"][metric][0] for row in case_outputs]
        right=[row["by_rotation"][metric][1] for row in case_outputs]
        lengths=[row["tokens"][metric] for row in case_outputs]
        raw_gaps=[gap for row in case_outputs
                  for gap in row["token_gaps_by_rotation"][metric]]
        if not all(math.isfinite(x) for x in vals+left+right+lengths+raw_gaps):
            hold("nonfinite article contrast")
        order=sorted(vals)
        trimmed=order[6:-6]
        loo=[(sum(vals)-v)/63 for v in vals]
        result[metric]={
          "mean":round(statistics.mean(vals),6),
          "median":round(statistics.median(vals),6),
          "positive":sum(v>0 for v in vals),
          "trimmed_mean_10pct":round(statistics.mean(trimmed),6),
          "leave_one_out_mean_bounds":[round(min(loo),6),round(max(loo),6)],
          "all_leave_one_out_means_positive":all(v>0 for v in loo),
          "descriptive_bootstrap_95pct":bootstrap_bounds(vals,model_name+":"+metric),
          "rotation0_mean":round(statistics.mean(left),6),
          "rotation1_mean":round(statistics.mean(right),6),
          "both_rotations_positive":sum(x>0 and y>0 for x,y in zip(left,right)),
          "rotation_sign_disagree":sum((x>0)!=(y>0) for x,y in zip(left,right)),
          "token_delta_mean":round(statistics.mean(lengths),3),
          "token_gap_absolute_mean":round(statistics.mean(abs(x) for x in raw_gaps),3),
          "token_gap_exact_matches":sum(x==0 for x in raw_gaps),
          "token_gap_comparisons":len(raw_gaps),
          "token_delta_spearman":spearman(vals,lengths),
          "n_articles":64
        }
    result["token_lengths"]={
       arm:{"mean":round(statistics.mean(x["mean_input_tokens"][arm]
                                         for x in case_outputs),2)}
       for arm in ARMS
    }
    return result


def source_inputs():
    roots={LABELS[0]:QROOT,LABELS[1]:MROOT,LABELS[2]:KROOT}
    verifiers={LABELS[0]:verify_q,LABELS[1]:verify_m,LABELS[2]:verify_k}
    programs={LABELS[0]:"exp053_agent_qwen_cpu.py",
              LABELS[1]:"exp053_agent_minicpm_cpu.py",
              LABELS[2]:"exp053_agent_k2_cpu.py"}
    cohort,cases,_=source_data()
    checkpoints=json.loads(FROZEN.read_bytes())["models"]
    runners={name:sha(Path(__file__).with_name(program).read_bytes())
             for name,program in programs.items()}
    for directory in roots.values():
        private_dir(directory)
        if list(directory.glob("failure-*.json")):
            hold("unreconciled case failure records")
        if len(list(directory.glob("case-*.json")))!=64:
            hold("wrong private case population")
    aggregates={
      LABELS[0]:QROOT/"summary-000-063-v2.json",
      LABELS[1]:MROOT/"summary-000-063-v1.json",
      LABELS[2]:KROOT/"summary-000-063-k2-v1.json",
    }
    output={name:[] for name in LABELS}
    input_hashes={}
    for name,path in aggregates.items():
        receipt=read_receipt(path)
        if receipt.get("case_count") != 64 or receipt.get("human_adjudicated") is not False:
            hold("invalid parent complete-cohort claim")
        input_hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    for i,(case_id,_) in enumerate(cases):
        original={}
        for name in LABELS:
            model=checkpoints[name]
            r=read_receipt(roots[name]/f"case-{i:03d}.json")
            verifiers[name](r,i,case_id,model["checkpoint_receipt_sha256"],
                            cohort["packet_sha256"],runners[name])
            if r.get("tokenizer_sha256") != model["tokenizer_sha256"]:
                hold("tokenizer SHA mismatch")
            if r.get("human_adjudicated") is not False:
                hold("cannot recast human evidence")
            original[name]=r
            output[name].append(per_case(r))
        case_compare(original[LABELS[0]],original[LABELS[1]])
        case_compare(original[LABELS[0]],original[LABELS[2]])
        case_compare(original[LABELS[1]],original[LABELS[2]])
    return output,input_hashes,runners


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--write-private",action="store_true")
    a=p.parse_args()
    all_rows,parents,runners=source_inputs()
    report={
      "schema":SCHEMA,
      "model_count":3,"case_count_per_model":64,
      "forwards_reused_per_model":384,
      "new_model_forwards":0,
      "human_adjudicated":False,
      "causal_grounding_proven":False,
      "pretraining_contamination_excluded":False,
      "source_arm_rotation_aligned":True,
      "input_full_cohort_sha256":parents,
      "runner_sha256":runners,
      "diagnostics":{m:diagnose(all_rows[m],m) for m in LABELS},
      "limitations":"Same frozen machine-selected exam wrong-option foils and source prompts. Rotation/length/leave-one-out checks are descriptive only; supported/decoy token gaps can cancel in the mean, so per-condition gap counts are authoritative. Token length is a confound proxy, not an independent causal control. No new foil, contamination, human adjudication, abstention, or generation data."
    }
    if a.write_private:
        private_dir(KROOT)
        digest=create_immutable(KROOT,"exp054-three-model-robustness-64-v1.json",report)
        print("EXP054_PRIVATE_AGGREGATE_COMPLETE sha256="+digest,flush=True)
    print("EXP054_SOURCE_ONLY_REUSE_PASS models=3 articles_per_model=64 new_forwards=0")
    print("EXP054_DESCRIPTIVE",report["diagnostics"],flush=True)


if __name__ == "__main__":
    main()
