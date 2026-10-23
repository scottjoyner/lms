#!/usr/bin/env python3
"""Source-only EXP-054 truncation severity audit, not evidence-sufficiency proof.

Reads frozen private source/case and token-bounded control identities. Produces
only aggregate fractions, with no licensed text, case IDs or model scores.
"""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from exp054_qwen_all_foils_cpu import load_frozen_inputs,RECEIPTS,PRECOMMIT
from exp053_agent_qwen_cpu import create_immutable,private_dir,read_receipt

SCHEMA="exp054-source-prefix-retention-descriptive-v1"


def hold(why):
    raise ValueError("HOLD EXP054 PREFIX RETENTION: "+why)


def analyze(lookup,source_cases):
    if not isinstance(lookup,dict) or len(lookup)!=64 or len(source_cases)!=64:
        hold("exact frozen cohort required")
    if sorted(lookup)!=[row.get("case_id") for row in source_cases]:
        hold("article population/order changed")
    fractions={"supported":[],"decoy":[]}
    lengths={"supported":[],"decoy":[]}
    for row in source_cases:
        supported=lookup[row["case_id"]]["article"]
        decoy=lookup[row["decoy_case_id"]]["article"]
        for arm,article,key in (
             ("supported",supported,"supported_prefix_codepoints"),
             ("decoy",decoy,"decoy_prefix_codepoints")):
            n=row[key]
            if (not isinstance(article,str) or not article or type(n)!=int or
                not 256<=n<=1600 or n>len(article)):
                hold("invalid frozen prefix/codepoint bounds")
            fractions[arm].append(n/len(article))
            lengths[arm].append(n)
        a,b=lengths["supported"][-1],lengths["decoy"][-1]
        if min(a,b)*1000<600*max(a,b):
            hold("minimum relative prefix ratio invalid")
    return {
      arm:{"average_retained_pct":round(100*statistics.mean(vals),3),
           "median_retained_pct":round(100*statistics.median(vals),3),
           "minimum_retained_pct":round(100*min(vals),3),
           "below_one_quarter_article":sum(v<.25 for v in vals),
           "below_half_article":sum(v<.5 for v in vals),
           "mean_prefix_codepoints":round(statistics.mean(lengths[arm]),2),
           "article_count":len(vals)}
      for arm,vals in fractions.items()
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-private",action="store_true")
    a=parser.parse_args()
    lookup,manifest,sources,foils,index=load_frozen_inputs()
    result={
       "schema":SCHEMA,
       "human_adjudicated":False,
       "semantic_answer_evidence_retained_verified":False,
       "model_forward_passes":0,
       "source_packet_sha256":manifest["packet_sha256"],
       "source_manifest_index_sha256":hashlib.sha256(
          (PRECOMMIT/"exp054-wordprefix-balanced-index-v1.json").read_bytes()).hexdigest(),
       "model_count":3,"case_count":64,
       "retention":analyze(lookup,sources),
       "limitations":"Unicode character retention is NOT a certified answer-evidence or semantic sufficiency measurement. It is not a token overlap, human judgment or new neural forward."
    }
    if a.write_private:
        private_dir(RECEIPTS)
        p=RECEIPTS/"exp054-source-prefix-retention-v1.json"
        encoded=(json.dumps(result,sort_keys=True,separators=(",",":"))+"\n").encode()
        if p.exists() or p.is_symlink():
            if read_receipt(p)!=result or p.read_bytes()!=encoded:
                hold("existing owner-private retention receipt not identical")
            h=hashlib.sha256(encoded).hexdigest()
            print("EXP054_RETENTION_AGGREGATE_REUSED_SHA256",h,flush=True)
        else:
            h=create_immutable(RECEIPTS,p.name,result)
            print("EXP054_RETENTION_AGGREGATE_NEW_SHA256",h,flush=True)
    print("EXP054_SOURCE_TRUNCATION_READONLY",result["retention"],flush=True)


if __name__=="__main__":
    main()
