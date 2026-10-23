#!/usr/bin/env python3
"""Full EXP-053 three-model paired MACHINE-ONLY source comparison.

Scientific human-adjudicated EXP-053 acceptance remains a separate HOLD.
Only reads owner-private immutable receipts; no model weights or dataset content
is printed, published, or modified.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

from exp053_agent_pair_compare import case_compare
from exp053_agent_qwen_cpu import (
    FROZEN, RECEIPTS as QROOT, private_dir, read_receipt, verify_case as verify_q,
    sha, create_immutable
)
from exp053_agent_minicpm_cpu import RECEIPTS as MROOT, verify_case as verify_m
from exp053_agent_k2_cpu import RECEIPTS as KROOT, verify_case as verify_k
from exp053_agent_qwen_summary import case_margins, source_data

LABELS = ("Qwen3.5-4B-Base", "MiniCPM-V-4.7-1B", "K2-Horizon-3.7B")
METRICS = ("D", "N", "S")
BOOTSTRAPS = 2000
SEED = 53053


def hold(reason):
    raise ValueError("HOLD EXP053 MACHINE-ONLY THREE-MODEL: " + reason)


def analyze(cases):
    if not 1 <= len(cases) <= 64:
        hold("invalid article count")
    # Synthetic tests may call with fewer than 64 cases; CLI demands all 64.
    result = {"models":{}, "paired_differences":{}}
    for metric in METRICS:
        for label in LABELS:
            values = [case[label][metric] for case in cases]
            if any(not math.isfinite(x) for x in values):
                hold("non-finite model log-probability margin")
            report = result["models"].setdefault(label,{})
            report[metric] = {
                "mean":round(statistics.mean(values),6),
                "positive":sum(x > 0 for x in values),
                "cases":len(values),
            }
        for a, b in ((LABELS[0],LABELS[1]),(LABELS[0],LABELS[2]),(LABELS[1],LABELS[2])):
            delta=[case[a][metric]-case[b][metric] for case in cases]
            rng=random.Random(SEED + sum(map(ord,a+b+metric)))
            draws=sorted(statistics.mean(delta[rng.randrange(len(delta))] for _ in delta)
                         for _ in range(BOOTSTRAPS))
            tag=a+" minus "+b
            result["paired_differences"].setdefault(tag,{})[metric]={
                "mean":round(statistics.mean(delta),6),
                "positive_cases":sum(v>0 for v in delta),
                "paired_article_bootstrap_95pct_descriptive":[
                    round(draws[int(BOOTSTRAPS*.025)],6),
                    round(draws[int(BOOTSTRAPS*.975)],6)],
                "cases":len(delta),
            }
    return result


def main():
    args=argparse.ArgumentParser(description=__doc__)
    args.add_argument("--write-private",action="store_true")
    opts=args.parse_args()
    roots={LABELS[0]:QROOT,LABELS[1]:MROOT,LABELS[2]:KROOT}
    verify={LABELS[0]:verify_q,LABELS[1]:verify_m,LABELS[2]:verify_k}
    for r in roots.values():
        private_dir(r)
        if list(r.glob("failure-*.json")):
            hold("unreconciled machine failure receipts")

    manifest,cases,_ = source_data()
    checkpoints=json.loads(FROZEN.read_bytes())["models"]
    sourcehashes={
        label:sha(Path(__file__).with_name({
          LABELS[0]:"exp053_agent_qwen_cpu.py",
          LABELS[1]:"exp053_agent_minicpm_cpu.py",
          LABELS[2]:"exp053_agent_k2_cpu.py"}[label]).read_bytes())
        for label in LABELS
    }
    rows=[]; file_hashes=[]
    for i,(cid,_) in enumerate(cases):
        result={}
        metadata={}
        for label in LABELS:
            path=roots[label] / ("case-%03d.json" % i)
            receipt=read_receipt(path)
            cfg=checkpoints[label]
            verify[label](receipt,i,cid,cfg["checkpoint_receipt_sha256"],
                          manifest["packet_sha256"],sourcehashes[label])
            if receipt.get("tokenizer_sha256") != cfg["tokenizer_sha256"]:
                hold("tokenizer identity drift")
            if receipt.get("condition_count") != 6:
                hold("missing source-arm/rotation coverage")
            result[label]=receipt
            metadata[label]=hashlib.sha256(path.read_bytes()).hexdigest()
        case_compare(result[LABELS[0]],result[LABELS[1]])
        case_compare(result[LABELS[0]],result[LABELS[2]])
        case_compare(result[LABELS[1]],result[LABELS[2]])
        rows.append({label:case_margins(result[label]) for label in LABELS})
        file_hashes.append({"index":i,"receipt_hashes":metadata})

    report={
        "schema":"exp053-three-native-checkpoint-machine-proxy-paired-v1",
        "complete_three_model_machine_only":True,
        "human_adjudicated":False,"causal_grounding_proven":False,
        "pretraining_contamination_excluded":False,
        "case_count":64,"forward_passes_per_model":384,
        "source_arm_rotation_parity_verified":True,
        "source_packet_sha256":manifest["packet_sha256"],
        "runner_sha256":sourcehashes,
        "model_checkpoint_receipt_sha256":{
            label:checkpoints[label]["checkpoint_receipt_sha256"] for label in LABELS},
        "article_receipts":file_hashes,
        "descriptive_statistics":analyze(rows),
        "limitations":"Automatic exam-key wrong-option proxy, no independent human plausibility validation. Descriptive paired bootstrap, native tokenizer differences and possible training contamination; no inference or semantic grounding promotion."
    }
    if not opts.write_private:
        print("EXP053_THREE_MACHINE_ONLY_PRIVATE_PREFLIGHT_PASS cases=64 per_model=384",flush=True)
        print("EXP053_THREE_DESCRIPTIVE",report["descriptive_statistics"],flush=True)
        return
    output_hash=create_immutable(KROOT,"paired-three-models-64case-v1.json",report)
    print("EXP053_THREE_MACHINE_ONLY_PRIVATE_PASS cases=64 total_forwards=1152 sha256="+output_hash,flush=True)
    print("EXP053_THREE_DESCRIPTIVE",report["descriptive_statistics"],flush=True)


if __name__ == "__main__":
    main()
