#!/usr/bin/env python3
"""Private, deterministic machine-proxy K2 receipt audit and descriptive statistics.

This does NOT generate model outputs, claim human annotation, or run the
preregistered independently human-adjudicated study.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from exp053_agent_k2_cpu import (FROZEN, RECEIPTS, ARM_ORDER, private_dir,
                                   read_receipt, verify_case, sha, create_immutable)
from exp053_race_review import _read_pinned_public_manifest


def fail(message):
    raise ValueError("HOLD EXP053 PRIVATE SUMMARY: " + message)


def source_data():
    cohort = _read_pinned_public_manifest()
    if len(cohort.get("rows", [])) != 64:
        fail("frozen source cohort unavailable")
    cases = sorted((x["case_id"],x["article_sha256"]) for x in cohort["rows"])
    if len({x[0] for x in cases}) != 64 or len({x[1] for x in cases}) != 64:
        fail("duplicate case/source identity")
    receipt = json.loads(FROZEN.read_bytes())
    model = receipt["models"]["K2-Horizon-3.7B"]
    return cohort,cases,model


def case_margins(row):
    conditions = row["conditions"]
    margins = {}
    for x in conditions:
        key = (x["arm"], x["rotation"])
        if key in margins or not isinstance(x["gold_logp"], float) or not isinstance(x["foil_logp"],float):
            fail("duplicate condition or nonfloat scores")
        if not all(math.isfinite(x[k]) and x[k] <= 0 for k in ("gold_logp","foil_logp")):
            fail("invalid model log probability")
        margins[key] = x["gold_logp"] - x["foil_logp"]
    required = {(arm,rotation) for rotation in (0,1) for arm in ARM_ORDER}
    if set(margins) != required:
        fail("six-source grid incomplete")
    avg={arm: statistics.mean(margins[(arm,r)] for r in (0,1)) for arm in ARM_ORDER}
    return {"D":avg["supported"]-avg["withheld"],
            "N":avg["decoy"]-avg["withheld"],
            "S":avg["supported"]-avg["decoy"]}


def summarize(records, *, count):
    if len(records) != count or count < 1 or count > 64:
        fail("noncontiguous or invalid case population")
    scores = {"D":[],"N":[],"S":[]}
    for row in records:
        values=case_margins(row)
        for key in scores:
            scores[key].append(values[key])
    return {key:{"mean":round(statistics.mean(v),6),
                 "median":round(statistics.median(v),6),
                 "positive":sum(x>0 for x in v),"n":len(v)}
            for key,v in scores.items()}



def verify_existing_aggregate(path, expected_result):
    if read_receipt(path) != expected_result:
        fail("existing aggregate content or receipt provenance differs")
    canonical=(json.dumps(expected_result,sort_keys=True,separators=(",",":"))+"\n").encode()
    if path.read_bytes() != canonical:
        fail("existing aggregate bytes differ from canonical source")
    return hashlib.sha256(canonical).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--count",type=int,required=True,help="contiguous prefix from 1 to 64")
    p.add_argument("--write-private",action="store_true",help="exclusive owner-private report")
    p.add_argument("--verify-existing",action="store_true",help="compare full canonical private aggregate without rewriting")
    a=p.parse_args()
    if a.write_private and a.verify_existing:
        fail("cannot create and verify existing report simultaneously")
    if not 1 <= a.count <= 64:
        fail("count must be in 1..64")
    private_dir(RECEIPTS)
    cohort,cases,model=source_data()
    runner_file=Path(__file__).with_name("exp053_agent_k2_cpu.py")
    runner_sha=sha(runner_file.read_bytes())
    rows=[];hashes=[]
    for i in range(a.count):
        file=RECEIPTS / ("case-%03d.json"%i)
        row=read_receipt(file)
        verify_case(row,i,cases[i][0],model["checkpoint_receipt_sha256"],
                    cohort["packet_sha256"],runner_sha)
        if row.get("tokenizer_sha256") != model["tokenizer_sha256"] or row.get("study_complete") is not False:
            fail("tokenizer identity or claim mismatch")
        if row.get("condition_count") != 6:
            fail("missing condition count")
        rows.append(row)
        hashes.append({"index":i,"sha256":hashlib.sha256(file.read_bytes()).hexdigest()})
    if any(RECEIPTS.glob("failure-*.json")):
        fail("failure receipts require reconciliation")
    result={"schema":"exp053-k2-agent-only-summary-v1",
            "human_adjudicated":False,"machine_selected_foil":True,
            "complete_frozen_cohort":a.count==64,
            "semantic_grounding_proven":False,
            "case_count":a.count,"condition_count":6*a.count,
            "model_checkpoint_receipt_sha256":model["checkpoint_receipt_sha256"],
            "runner_sha256":runner_sha,
            "receipts":hashes,
            "descriptive_metrics":summarize(rows,count=a.count),
            "limitation":"Machine exam-key foil, no independently adjudicated plausible foil, no statistical inference or causal claims."}
    name="summary-000-%03d-k2-v1.json"%(a.count-1)
    if a.verify_existing:
        digest=verify_existing_aggregate(RECEIPTS / name, result)
        print("EXP053_K2_PRIVATE_SUMMARY_REUSED_VERIFIED cases=%d rows=%d sha256=%s" %
              (a.count,6*a.count,digest),flush=True)
        return
    if not a.write_private:
        print("EXP053_K2_PRIVATE_SUMMARY_PREFLIGHT_PASS cases=%d rows=%d output=none" %
              (a.count,6*a.count))
        return
    checksum=create_immutable(RECEIPTS,name,result)
    print("EXP053_K2_PRIVATE_SUMMARY_WRITTEN cases=%d rows=%d sha256=%s" %
          (a.count,6*a.count,checksum))


if __name__=="__main__":
    main()
