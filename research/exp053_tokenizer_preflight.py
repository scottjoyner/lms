#!/usr/bin/env python3
"""Tokenizer-only, source-custody-controlled research feasibility; zero forwards."""
import argparse,hashlib,json,os,statistics
from pathlib import Path
from exp053_race_review import (MANIFEST,MANIFEST_SHA,PROTOCOL,PROTOCOL_SHA,
                                SOURCE,SOURCE_SHA,digest,filehash,deny)
os.environ.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",TOKENIZERS_PARALLELISM="false")
ROOT=Path(__file__).resolve().parent
PRIVATE=Path("/media/scott/SSD_4TB/exp053-private-20261016/packet.json")
SPEC=ROOT/"EXP053-TOKENIZER-PREFLIGHT-SPEC-2026-10-16.md"
SPEC_SHA="92643aa9e0603d29eeab8dfd24680615036b7ee49d88e12d439a8e5c569b8fb7"
PLAN=ROOT/"EXP053-MEASUREMENT-PLAN-2026-10-16.md"
PLAN_SHA="5239be2dd2413af98759a2ac5fdde100f2407d2de07149347f272d1e5ee0922d"
PATHS={
  "K2-Horizon-3.7B":Path("/home/scott/models/research/K2-Horizon-3.7B"),
  "Qwen3.5-4B-Base":Path("/home/scott/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B-Base/snapshots/1001bb4d826a52d1f399e183466143f4da7b741b"),
  "MiniCPM-V-4.7-1B":Path("/media/scott/SSD_4TB/hf-cache/huggingface/hub/models--openbmb--MiniCPM-V-4.7-1B/snapshots/286817f1132b998913a129a221c89b4402b4ab80")}
SOURCE_ARMS=("supported","withheld","decoy")
LETTERS="ABCD"
MAX_TOKENS=4096
WITHHELD="The source passage is withheld."

def validate_custody():
    for p,expected in ((SOURCE,SOURCE_SHA),(PROTOCOL,PROTOCOL_SHA),
                       (MANIFEST,MANIFEST_SHA),(SPEC,SPEC_SHA),(PLAN,PLAN_SHA)):
        if not p.is_file() or filehash(p)!=expected:deny("tokenizer preflight authority drift "+str(p))
    manifest=json.loads(MANIFEST.read_text())
    private=json.loads(PRIVATE.read_text())
    if digest(private)!=manifest["packet_sha256"]:deny("private packet mismatch")
    cases=private["cases"]
    if len(cases)!=64:deny("expected 64 original candidate source papers")
    lookup={x["case_id"]:x for x in cases}
    if len(lookup)!=64 or len(set(x["article"] for x in cases))!=64:
        deny("repeated article, case ID, or source text")
    for m in manifest["rows"]:
        c=lookup.get(m["case_id"])
        if c is None or hashlib.sha256(c["article"].encode()).hexdigest()!=m["article_sha256"]:
            deny("source paper hash mismatch")
    return lookup,manifest

def options_for(case,rotation):
    values=[case["options"][x] for x in LETTERS]
    if rotation==1:values=values[2:]+values[:2]
    elif rotation!=0:deny("unknown label rotation")
    return dict(zip(LETTERS,values))

def prompt(question,options,passage):
    return ("Read ONLY the passage and answer the multiple-choice question. Choose the\n"
            "single option best supported by the passage. If the passage is insufficient,\n"
            "output UNKNOWN. Write exactly one answer letter if supported.\n"
            "Passage:\n"+passage+"\nQuestion: "+question+"\n"+
            "".join(f"{l}) {options[l]}\n" for l in LETTERS)+"Answer:")

def prompts(lookup):
    ids=sorted(lookup)
    for i,cid in enumerate(ids):
        case=lookup[cid]
        for r in (0,1):
            opts=options_for(case,r)
            for arm in SOURCE_ARMS:
                passage=(case["article"] if arm=="supported" else
                          WITHHELD if arm=="withheld" else lookup[ids[(i+1)%64]]["article"])
                yield cid,r,arm,prompt(case["question"],opts,passage)

def percent(values,p):
    v=sorted(values)
    return v[int((len(v)-1)*p)]

def preflight():
    lookup,manifest=validate_custody()
    cases=list(prompts(lookup))
    if len(cases)!=384:deny("incomplete candidate-only source/rotation grid")
    from transformers import AutoTokenizer
    models={}
    for name,p in PATHS.items():
        for fn in ("config.json","tokenizer_config.json","tokenizer.json"):
            if not (p/fn).is_file():deny("missing local tokenizer file "+name+"/"+fn)
        tok=AutoTokenizer.from_pretrained(str(p),local_files_only=True,
                     trust_remote_code=(name=="K2-Horizon-3.7B"))
        labels={letter:tok.encode(" "+letter,add_special_tokens=False)
                for letter in LETTERS}
        if any(len(ids)!=1 for ids in labels.values()):
            deny(name+" multi-token answer label")
        if len(set(ids[0] for ids in labels.values()))!=4:
            deny(name+" nonunique label IDs")
        lengths=[];checks=0
        for cid,rot,arm,s in cases:
            base=tok.encode(s,add_special_tokens=True)
            if not 1<=len(base)<=MAX_TOKENS:
                deny(name+" 4096-token window exceeded")
            lengths.append(len(base))
            for letter in LETTERS:
                combined=tok.encode(s+" "+letter,add_special_tokens=True)
                if combined!=base+labels[letter]:
                    deny(name+" prefix/continuation BPE mismatch case="+cid+
                         " arm="+arm+" rot="+str(rot)+" label="+letter)
                checks+=1
        models[name]={"config_sha256":filehash(p/"config.json"),
                      "tokenizer_json_sha256":filehash(p/"tokenizer.json"),
                      "tokenizer_config_sha256":filehash(p/"tokenizer_config.json"),
                      "checkpoint_directory":p.name,
                      "tokenizer_class":type(tok).__name__,
                      "candidate_answer_token_ids":{k:v[0] for k,v in labels.items()},
                      "source_prompts_tokenized":len(lengths),
                      "label_boundary_checks_passed":checks,
                      "prompt_tokens_min":min(lengths),
                      "prompt_tokens_median":statistics.median(lengths),
                      "prompt_tokens_p95":percent(lengths,.95),
                      "prompt_tokens_max":max(lengths),
                      "model_forward_passes":0}
        del tok
        print("TOKENIZER_ONLY_OK",name,"prompts",len(lengths),
              "boundaries",checks,"max_input_tokens",max(lengths),flush=True)
    return {"schema":"exp053-tokenizer-only-preflight-v1",
            "purpose":"source-only prospective feasibility, no model inference",
            "spec_sha256":SPEC_SHA,"manifest_sha256":MANIFEST_SHA,
            "source_sha256":SOURCE_SHA,"measurement_plan_sha256":PLAN_SHA,
            "candidate_articles":len(lookup),"source_conditions":len(SOURCE_ARMS),
            "label_rotations":2,"candidate_prompt_count_per_model":len(cases),
            "human_reviewed_articles":0,"model_forward_passes":0,
            "approved_only_final_decoy_checked":False,
            "pretraining_contamination_excluded":False,
            "models":models}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--execute-tokenizers",action="store_true")
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    validate_custody()
    if not args.execute_tokenizers:
        print("DRY_RUN_NO_TOKENIZER_OR_MODEL",SPEC_SHA);return
    if args.out is None or args.out.parent.resolve()!=ROOT.resolve() or args.out.exists():
        deny("output must be new source-free file in research directory")
    result=preflight()
    with args.out.open("x") as f:
        f.write(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("SUCCESS_SOURCE_ONLY_0_FORWARDS",len(result["models"]))
if __name__=="__main__":main()
