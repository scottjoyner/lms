#!/usr/bin/env python3
"""EXP-054 Qwen pilot: all predeclared wrong exam foils, token-gap-bounded sources.

Single CPU forward scores all four possible answer-letter continuations; no
human plausibility or semantic evidence claim. Exact source-only controls must
have been frozen before this file is executed. New immutable private receipts.
"""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time

from exp053_agent_only import machine_receipt
from exp053_tokenizer_preflight import PRIVATE, PATHS, validate_custody
from exp053_agent_qwen_cpu import (
    OUT_ROOT, FROZEN, sha, digest_file, private_dir, read_receipt,
    create_immutable,
)
from exp053_k2_resource_gate import probe,inspect as pressure_inspect
from exp054_prefix_length_control import prompt,options_for,digest
from exp054_word_boundary_token_balance import (
    SCHEMA as SOURCE_SCHEMA, canonical_bytes,root_manifest,MODELS
)

NAME="Qwen3.5-4B-Base"
SCHEMA="exp054-qwen-three-foil-bounded-source-case-v1"
RECEIPTS=OUT_ROOT/"exp054-qwen-three-foil-20261023-v1"
PRECOMMIT=OUT_ROOT/"k2-machine-proxy-20261022-v1"
ARMS=("supported_prefix","decoy_prefix")


def hold(reason):
    raise ValueError("HOLD EXP054 QWEN PILOT: "+reason)


def load_frozen_inputs():
    lookup,manifest=validate_custody()
    admission=machine_receipt(PRIVATE.parent)
    if set(lookup)!=set(admission["accepted"]) or len(lookup)!=64:
        hold("source or admission case mismatch")
    pinned=read_receipt(PRECOMMIT/"exp054-wordprefix-balanced-index-v1.json")
    code_sha=sha(Path(__file__).with_name("exp054_word_boundary_token_balance.py").read_bytes())
    if pinned.get("algorithm_source_sha256")!=code_sha or pinned.get("source_packet_sha256")!=manifest["packet_sha256"]:
        hold("frozen source/control code hash drift")
    foil_path=PRECOMMIT/"exp054-all-three-machine-foils-v1.json"
    foils=read_receipt(foil_path)
    if (foils.get("source_packet_sha256")!=manifest["packet_sha256"] or
        foils.get("based_on_target_model_scores") is not False or
        foils.get("human_adjudicated") is not False or
        sha(foil_path.read_bytes())!=pinned.get("predeclared_machine_foils_sha256")):
        hold("foil precommit identity mismatch")
    if [x["case_id"] for x in foils["cases"]] != sorted(lookup):
        hold("wrong-option roster case order differs")
    conditions=[]
    for i,part in enumerate(pinned.get("parts",[])):
        name=f"exp054-wordprefix-balanced-part{i:02}-v1.json"
        if part.get("filename")!=name or part.get("starting_index")!=8*i:
            hold("manifest shard ordering mismatch")
        path=PRECOMMIT/name
        if sha(path.read_bytes())!=part.get("sha256"):
            hold("manifest shard bytes changed")
        source=read_receipt(path)
        if source.get("algorithm_source_sha256")!=code_sha or source.get("source_packet_sha256")!=manifest["packet_sha256"]:
            hold("manifest shard authority drift")
        conditions.extend(source["cases"])
    if len(conditions)!=64 or pinned.get("case_count")!=64:
        hold("not all 64 frozen source cases present")
    report=root_manifest(conditions,manifest["packet_sha256"],code_sha)
    report["predeclared_machine_foils_sha256"]=sha(foil_path.read_bytes())
    if sha(canonical_bytes(report))!=pinned.get("complete_manifest_sha256"):
        hold("frozen source control total hash mismatch")
    ids=sorted(lookup)
    if [c["case_id"] for c in conditions]!=ids:
        hold("source case ordering differs")
    return lookup,manifest,conditions,foils,pinned


def prepare_case(case,source,foil,tok,label_ids):
    cid=case["case_id"]
    if cid!=source["case_id"] or cid!=foil["case_id"]:
        hold("source and foil identity mismatch")
    other=source["decoy_case_id"]
    if source["cohort_index"] is None:
        hold("source index missing")
    article=case["article"]
    # Caller provides decoy source separately; this function only consumes
    # the already frozen, reconstructed strings.
    gold=foil["gold_label_private"]
    wrong=foil["all_three_machine_wrong_labels"]
    if (gold not in "ABCD" or sorted(wrong)!=sorted([l for l in "ABCD" if l!=gold])
        or foil["exp053_original_machine_foil"] not in wrong):
        hold("machine-only wrong options mismatch")
    if not isinstance(article,str) or not article:
        hold("source article empty")
    rows=[]
    for control in source["conditions"]:
        rotation=control["rotation"]
        arm=control["arm"]
        if rotation not in (0,1) or arm not in ARMS:
            hold("invalid source/rotation")
        rows.append((rotation,arm,control))
    if [(a,r) for r,a,_ in rows] != [(a,r) for r in (0,1) for a in ARMS]:
        hold("source condition order")
    return rows


def rotate(label,rotation):
    return "ABCD"[("ABCD".index(label)+2*rotation)%4]


def check_receipt(value,index,cid,packet_sha,checkpoint_sha,runner_sha,source_sha,foil_sha):
    if (value.get("schema")!=SCHEMA or value.get("cohort_index")!=index or
        value.get("case_id")!=cid or
        value.get("private_packet_sha256")!=packet_sha or
        value.get("checkpoint_receipt_sha256")!=checkpoint_sha or
        value.get("runner_sha256")!=runner_sha or
        value.get("source_control_index_sha256")!=source_sha or
        value.get("frozen_three_foils_sha256")!=foil_sha or
        value.get("human_adjudicated") is not False or
        value.get("model_forward_passes")!=4):
        hold("immutable prior pilot receipt identity mismatch")
    conditions=value.get("conditions",[])
    if len(conditions)!=4 or [(x.get("arm"),x.get("rotation")) for x in conditions] != [
       (arm,rot) for rot in (0,1) for arm in ARMS]:
        hold("prior pilot conditions missing")
    for x in conditions:
        scores=x.get("logprob_by_label")
        if (not isinstance(scores,dict) or set(scores)!=set("ABCD") or
            any(type(z)!=float or not math.isfinite(z) or z>0 for z in scores.values()) or
            x.get("gold_label") not in "ABCD" or
            set(x.get("foil_labels",[]))!=set("ABCD")-{x.get("gold_label")}):
            hold("prior pilot contains invalid model scores")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start",type=int,default=0)
    p.add_argument("--count",type=int,default=1)
    p.add_argument("--execute-source-backed",action="store_true")
    a=p.parse_args()
    if a.start<0 or a.count<1 or a.start+a.count>64:
        hold("invalid bounded source selection")
    if os.geteuid()==0 or os.environ.get("HF_HUB_OFFLINE")!="1" or os.environ.get("TRANSFORMERS_OFFLINE")!="1":
        hold("nonroot fully offline execution required")
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        hold("GPU execution disabled")
    private_dir(OUT_ROOT)
    lookup,manifest,source_cases,foils,pinned=load_frozen_inputs()
    checkpoint=json.loads(FROZEN.read_bytes())["models"][NAME]
    model_dir=PATHS[NAME]
    if sha((model_dir/"config.json").read_bytes())!=checkpoint["config_sha256"]:
        hold("model config SHA drift")
    if sha((model_dir/"tokenizer.json").read_bytes())!=checkpoint["tokenizer_sha256"]:
        hold("model tokenizer SHA drift")
    for shard in checkpoint["shards"]:
        path=model_dir/shard["filename"]
        if not path.is_file() or path.stat().st_size != shard["bytes"] or digest_file(path)!=shard["sha256"]:
            hold("BF16 shard identity drift")
    original_ids=sorted(lookup)
    source_sha=sha((PRECOMMIT/"exp054-wordprefix-balanced-index-v1.json").read_bytes())
    foil_sha=sha((PRECOMMIT/"exp054-all-three-machine-foils-v1.json").read_bytes())
    runner_sha=sha(Path(__file__).read_bytes())
    if not a.execute_source_backed:
        print("EXP054_QWEN_THREE_FOILS_SOURCE_PREFLIGHT_PASS",{
            "case_count":a.count,"new_forwards":0,
            "model":"exact Qwen3.5-4B-Base BF16",
            "source_control_frozen":True,"alternative_foils_frozen":True},flush=True)
        return
    RECEIPTS.mkdir(mode=0o700,exist_ok=True)
    private_dir(RECEIPTS)
    pending=[]
    for i in range(a.start,a.start+a.count):
        cid=original_ids[i]
        path=RECEIPTS/f"case-{i:03d}.json"
        if path.exists() or path.is_symlink():
            check_receipt(read_receipt(path),i,cid,manifest["packet_sha256"],
                          checkpoint["checkpoint_receipt_sha256"],runner_sha,source_sha,foil_sha)
            print("EXP054_QWEN_EXISTING_CASE_VERIFIED",i,flush=True)
        else:
            pending.append((i,cid))
    if not pending:
        print("EXP054_QWEN_SELECTED_CASES_ALREADY_VERIFIED",flush=True)
        return
    before=probe()
    time.sleep(4)
    checks,stats=pressure_inspect(before,probe(),4)
    if not all(checks.values()):
        hold("host resource pressure gate refused model load "+",".join(k for k,v in checks.items() if not v))
    print("EXP054_QWEN_HOST_PRESSURE_GO",stats,flush=True)
    import torch
    from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
    if str(torch.__version__)!="2.6.0+cpu" or importlib.metadata.version("transformers")!="5.20.1":
        hold("unqualified isolated CPU model runtime")
    tok=AutoTokenizer.from_pretrained(str(model_dir),local_files_only=True,trust_remote_code=False)
    label_ids={l:tok.encode(" "+l,add_special_tokens=False) for l in "ABCD"}
    if any(len(v)!=1 for v in label_ids.values()) or len({v[0] for v in label_ids.values()})!=4:
        hold("answer labels are not four unique one-token continuations")
    prepared={}
    for i,cid in pending:
        case=lookup[cid]
        frozen=source_cases[i]
        foil=foils["cases"][i]
        rows=prepare_case(case,frozen,foil,tok,label_ids)
        batch=[]
        for rot,arm,control in rows:
            length=frozen["supported_prefix_codepoints" if arm=="supported_prefix" else "decoy_prefix_codepoints"]
            text=case["article"][:length] if arm=="supported_prefix" else lookup[frozen["decoy_case_id"]]["article"][:length]
            s=prompt(case["question"],options_for(case,rot),text)
            if digest(s)!=control["prompt_sha256"] or digest(text)!=control["passage_sha256"]:
                hold("frozen scored prompt bytes disagree")
            tokens=tok.encode(s,add_special_tokens=True)
            if len(tokens)!=control["input_tokens"][NAME] or len(tokens)>4096:
                hold("tokenization differs from frozen source")
            for l in "ABCD":
                if tok.encode(s+" "+l,add_special_tokens=True)!=tokens+label_ids[l]:
                    hold("single-token label boundary changed")
            batch.append((rot,arm,control["prompt_sha256"],tokens))
        prepared[cid]=batch
    torch.set_num_threads(2)
    model=Qwen3_5ForConditionalGeneration.from_pretrained(
        str(model_dir),local_files_only=True,trust_remote_code=False,
        dtype=torch.bfloat16,attn_implementation="eager")
    model.eval()
    runtime={"python":sys.version.split()[0],"torch":str(torch.__version__),
             "transformers":importlib.metadata.version("transformers"),
             "model_source_sha256":digest_file(Path(inspect.getfile(Qwen3_5ForConditionalGeneration)))}
    print("EXP054_QWEN_MODEL_LOADED_CPU pending_cases="+str(len(pending)),flush=True)
    for i,cid in pending:
        frozen_foil=foils["cases"][i]
        recorded=[]
        try:
            for rot,arm,psha,tokens in prepared[cid]:
                with torch.inference_mode():
                    result=model(input_ids=torch.tensor([tokens],dtype=torch.long),use_cache=False)
                    vector=result.logits[0,-1,:].float()
                    if not bool(torch.isfinite(vector).all().item()):
                        hold("model logits nonfinite")
                    logp=torch.log_softmax(vector,dim=-1)
                    by_label={letter:float(logp[label_ids[letter][0]].item()) for letter in "ABCD"}
                if any(not math.isfinite(v) or v>0 for v in by_label.values()):
                    hold("invalid answer probability")
                gold=rotate(frozen_foil["gold_label_private"],rot)
                wrong=[rotate(w,rot) for w in frozen_foil["all_three_machine_wrong_labels"]]
                recorded.append({"rotation":rot,"arm":arm,"prompt_sha256":psha,
                                 "input_tokens":len(tokens),"gold_label":gold,
                                 "foil_labels":wrong,"logprob_by_label":by_label})
        except Exception as exc:
            create_immutable(RECEIPTS,f"failure-{i:03}-{time.time_ns()}.json",{
                "schema":"exp054-qwen-private-failure-v1","cohort_index":i,
                "failure_type":type(exc).__name__,"human_adjudicated":False,
                "status":"INCOMPLETE_NO_SCIENTIFIC_RESULT"})
            raise
        payload={
           "schema":SCHEMA,"cohort_index":i,"case_id":cid,
           "human_adjudicated":False,"semantic_evidence_preserved_verified":False,
           "machine_only":True,"model_forward_passes":4,
           "private_packet_sha256":manifest["packet_sha256"],
           "checkpoint_receipt_sha256":checkpoint["checkpoint_receipt_sha256"],
           "tokenizer_sha256":checkpoint["tokenizer_sha256"],
           "runner_sha256":runner_sha,
           "source_control_index_sha256":source_sha,
           "frozen_three_foils_sha256":foil_sha,
           "runtime":runtime,"conditions":recorded,
           "limitations":"Frozen wrong exam-answer labels, word-prefix length tolerance, no independent human adjudication or proof that prefix includes the answer evidence."}
        checksum=create_immutable(RECEIPTS,f"case-{i:03d}.json",payload)
        print("EXP054_QWEN_PRIVATE_ALL_FOILS_PILOT_OK case_index=%d conditions=4 sha256=%s"%
              (i,checksum),flush=True)


if __name__=="__main__":
    try:
        main()
    except Exception as exc:
        print("EXP054_QWEN_PILOT_HOLD",type(exc).__name__,flush=True)
        raise SystemExit(2)
