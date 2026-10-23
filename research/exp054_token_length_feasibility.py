#!/usr/bin/env python3
"""EXP-054 offline tokenizer-only feasibility for the frozen codepoint-paired controls.

Fail-closed on source/custody/prompt drift. No model weights or inference; no
licensed text, answer keys, or individual token IDs leave the private container.
"""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

from exp054_prefix_length_control import (
    build_manifest, make_pair, canonical_bytes, verify_parts, RECEIPTS,
    private_dir, digest, options_for, prompt
)
from exp053_agent_only import machine_receipt
from exp053_tokenizer_preflight import PRIVATE, PATHS, validate_custody
from exp053_agent_qwen_cpu import FROZEN, read_receipt, create_immutable

MODELS=("Qwen3.5-4B-Base","MiniCPM-V-4.7-1B","K2-Horizon-3.7B")
LABELS="ABCD"
SCHEMA="exp054-tokenizer-only-prefix-feasibility-v1"
MAX_PROMPT_TOKENS=4096


def hold(message):
    raise ValueError("HOLD EXP054 TOKEN LENGTH: "+message)


def check_frozen_plan():
    lookup,manifest=validate_custody()
    admitted=machine_receipt(PRIVATE.parent)
    expected=build_manifest(lookup,admitted,manifest["packet_sha256"])
    path=RECEIPTS/"exp054-prefix-length-control-index-v1.json"
    index=read_receipt(path)
    frozen_code=hashlib.sha256(Path(__file__).with_name("exp054_prefix_length_control.py").read_bytes()).hexdigest()
    if index.get("preflight_code_sha256")!=frozen_code:
        hold("length-control generator source changed after freeze")
    # The original frozen manifest includes the source code pin itself.
    # Reconstruct the same canonical object, not merely its source rows.
    expected["preflight_code_sha256"]=frozen_code
    if not verify_parts(index,expected):
        hold("source manifest shards did not replay")
    if index.get("source_case_count")!=64:
        hold("unexpected source population")
    return lookup, expected, hashlib.sha256(path.read_bytes()).hexdigest()


def enumerate_prompts(lookup,expected):
    rows=[]
    for case in expected["cases"]:
        cid=case["case_id"]
        supported=lookup[cid]
        decoy=lookup[case["decoy_case_id"]]
        pair=make_pair(supported,decoy)
        if pair["codepoints"]!=case["codepoint_budget"]:
            hold("frozen character budget drift")
        per_case=[]
        planned={(x["arm"],x["rotation"]):x for x in case["controls"]}
        for rotation in (0,1):
            choices=options_for(supported,rotation)
            for arm in ("supported_prefix","decoy_prefix"):
                s=prompt(supported["question"],choices,pair[arm])
                if digest(s)!=planned[arm,rotation]["prompt_sha256"]:
                    hold("prompt source digest drift")
                per_case.append((rotation,arm,s))
        if len(per_case)!=4:
            hold("incomplete prompt grid")
        rows.append((cid,per_case))
    if len(rows)!=64:
        hold("not all source cases reconstructed")
    return rows


def token_counts(tokenizer,prompts):
    label_ids={letter:tokenizer.encode(" "+letter,add_special_tokens=False)
               for letter in LABELS}
    if any(len(v)!=1 for v in label_ids.values()) or len({v[0] for v in label_ids.values()})!=4:
        hold("label must be distinct one-token continuation")
    records=[];all_deltas=[];all_absolute=[]
    for cid,rows in prompts:
        counts={}
        for rotation,arm,text in rows:
            tokens=tokenizer.encode(text,add_special_tokens=True)
            if len(tokens)<1 or len(tokens)>MAX_PROMPT_TOKENS:
                hold("prompt outside source token window")
            for label in LABELS:
                if tokenizer.encode(text+" "+label,add_special_tokens=True)!=tokens+label_ids[label]:
                    hold("label-BPE concatenation mismatch")
            counts[arm,rotation]=len(tokens)
        deltas=[]
        for r in (0,1):
            diff=counts["supported_prefix",r]-counts["decoy_prefix",r]
            deltas.append(diff)
            all_deltas.append(diff)
            all_absolute.append(abs(diff))
        records.append({"case_id":cid,
                        "supported_tokens":[counts["supported_prefix",0],counts["supported_prefix",1]],
                        "decoy_tokens":[counts["decoy_prefix",0],counts["decoy_prefix",1]],
                        "signed_token_gap":deltas})
    if len(all_deltas)!=128:
        hold("token comparison grid incomplete")
    return {
      "total_prompt_count":256,"label_boundary_checks":1024,
      "prompt_length_max":max(x for row in records for x in row["supported_tokens"]+row["decoy_tokens"]),
      "exact_token_matches":sum(v==0 for v in all_deltas),
      "absolute_gap_mean":round(statistics.mean(all_absolute),3),
      "absolute_gap_median":round(statistics.median(all_absolute),3),
      "absolute_gap_max":max(all_absolute),
      "within_8_tokens":sum(v<=8 for v in all_absolute),
      "within_16_tokens":sum(v<=16 for v in all_absolute),
      "within_32_tokens":sum(v<=32 for v in all_absolute),
      "condition_count":len(all_deltas),
      "rows":records
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--write-private",action="store_true")
    args=p.parse_args()
    lookup,plan,index_sha=check_frozen_plan()
    prompts=enumerate_prompts(lookup,plan)
    from transformers import AutoTokenizer
    frozen_models=json.loads(FROZEN.read_bytes())["models"]
    summaries={}
    for name in MODELS:
        directory=PATHS[name]
        tok=directory/"tokenizer.json"
        if hashlib.sha256(tok.read_bytes()).hexdigest()!=frozen_models[name]["tokenizer_sha256"]:
            hold("checkpoint tokenizer SHA mismatch")
        tokenizer=AutoTokenizer.from_pretrained(str(directory),
                            local_files_only=True,trust_remote_code=False)
        stats=token_counts(tokenizer,prompts)
        summaries[name]={
            "tokenizer_class":type(tokenizer).__name__,
            "tokenizer_sha256":frozen_models[name]["tokenizer_sha256"],
            **stats
        }
        print("EXP054_PREFIX_TOKENIZER",name,{
            k:v for k,v in stats.items() if k!="rows"
        },flush=True)
        del tokenizer
    report={"schema":SCHEMA,"source_length_index_sha256":index_sha,
            "model_count":3,"articles":64,"prompts_per_model":256,
            "human_adjudicated":False,"model_forward_passes":0,
            "token_length_parity_verified":all(
                v["exact_token_matches"]==128 for v in summaries.values()),
            "source_evidence_retained_verified":False,
            "native_tokenizer_results":summaries,
            "limitations":"Equal Unicode codepoint prefixes are not equal token lengths; no semantic evidence check or model inference was done."}
    if args.write_private:
        private_dir(RECEIPTS)
        # Do not weaken the 32 KiB private writer; case rows remain private,
        # summarized in this first feasibility version without per-case disclosure.
        canonical={**report,
            "native_tokenizer_results":{
                model:{k:v for k,v in stat.items() if k!="rows"}
                for model,stat in summaries.items()}}
        digest_value=create_immutable(RECEIPTS,"exp054-prefix-token-feasibility-v1.json",canonical)
        print("EXP054_PRIVATE_TOKEN_FEASIBILITY_FROZEN sha256="+digest_value,flush=True)
    print("EXP054_TOKEN_ONLY_FEASIBILITY_DONE new_model_forwards=0",flush=True)


if __name__=="__main__":
    main()
