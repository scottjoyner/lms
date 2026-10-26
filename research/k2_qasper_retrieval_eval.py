#!/usr/bin/env python3
"""EXP046: full-paper paragraph BM25 retrieval and K2 source answering."""
import os
for k,v in {"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","CUDA_VISIBLE_DEVICES":"-1",
 "TOKENIZERS_PARALLELISM":"false","OMP_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"1",
 "MKL_NUM_THREADS":"2"}.items():os.environ[k]=v
import argparse,gc,hashlib,time
from pathlib import Path
import torch
from transformers import StoppingCriteriaList
from k2_interpolation_pilot import (MID,RL,TOK_SHA,IDX_SHA,CONFIG_SHA,
 preflight,check_shard_layout,sha,available_gib)
from k2_first_newline_stopping import StopAtGeneratedNewline
from k2_qasper_retrieval_fixture import (
 examples,prompt,receipt,aggregate,score,retrieval_audit,KINDS as CONDITIONS)
from k2_qasper_evidence_fixture import SOURCE_SHA
from k2_exp046_reporting import report,hit_conditioned

ROOT=Path("/home/scott/git/wt-knowledge-k2-latex-20261014/50-Research/llm-research/latent-world-geometry-2026")
PROTOCOL_SHA="880f14d3fbfcddb322ac184a7525a95f3a0e8231cd4304bca21100b014a9c256"
COHORT_SHA="f44ae4e5dcaf8f0b8455ee8f89f00207f8428461c5f536937ef11db01a2998f5"
MAX_NEW=16
MODES=("default","yarn")
torch.set_num_threads(2)

def frozen_sources():
    if sha(ROOT/"EXPERIMENT-046-PROTOCOL-2026-10-15.md")!=PROTOCOL_SHA:
        raise RuntimeError("HOLD protocol SHA mismatch")
    for i,root in enumerate((MID,RL)):
        if not root.is_dir() or root.is_symlink():
            raise RuntimeError("HOLD model checkpoint path")
        if sha(root/"model.safetensors.index.json")!=IDX_SHA[i]:
            raise RuntimeError("HOLD weight index")
        if sha(root/"config.json")!=CONFIG_SHA[i]:
            raise RuntimeError("HOLD config")
        if sha(root/"tokenizer.json")!=TOK_SHA:
            raise RuntimeError("HOLD tokenizer")
    cases,retrieval=examples()
    if len(cases)!=60 or receipt(cases,retrieval)!=COHORT_SHA:
        raise RuntimeError("HOLD QASPER retrieval cohort drift")
    return cases,retrieval

def generate_one(model,tok,row):
    if available_gib()<22:raise RuntimeError("HOLD in-run RAM <22GiB")
    inputs=tok(prompt(row),return_tensors="pt",truncation=False,add_special_tokens=True)
    inputs.pop("token_type_ids",None)
    n=int(inputs["input_ids"].shape[1])
    if not 20<n<750:raise RuntimeError("HOLD prompt too long")
    pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    stopper=StopAtGeneratedNewline(tok,n)
    with torch.inference_mode():
        out=model.generate(**inputs,do_sample=False,num_beams=1,
          max_new_tokens=MAX_NEW,use_cache=True,
          stopping_criteria=StoppingCriteriaList([stopper]),
          pad_token_id=pad,eos_token_id=tok.eos_token_id,
          return_dict_in_generate=True,output_scores=True)
    seq=out.sequences
    if seq.ndim!=2 or seq.shape[0]!=1:
        raise RuntimeError("HOLD generation topology")
    toks=seq[0,n:].tolist()
    if not 1<=len(toks)<=MAX_NEW or len(out.scores)!=len(toks):
        raise RuntimeError("HOLD generator length")
    if any(not torch.isfinite(z).all().item() for z in out.scores):
        raise RuntimeError("HOLD nonfinite generation scores")
    eos=tok.eos_token_id
    eosids=set(eos) if isinstance(eos,(list,tuple)) else {eos}
    stop=next((i for i,v in enumerate(toks) if v in eosids),len(toks))
    decoded=tok.decode(toks[:stop],skip_special_tokens=True)
    newline="\n" in decoded
    if newline!=bool(stopper.hit):
        raise RuntimeError("HOLD newline stopping mismatch")
    visible=decoded.split("\n",1)[0] if newline else decoded
    eos_found=stop<len(toks)
    capped=stop>=MAX_NEW and not newline and not eos_found
    result=score(row,visible,hit_cap=capped,hit_newline=newline)
    result["native_eos"]=int(eos_found)
    return result,n

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute",action="store_true")
    ap.add_argument("--out",type=Path)
    args=ap.parse_args()
    rows,retrieval=frozen_sources()
    print("EXP046_PREFLIGHT_OK",len(rows),"cohort",COHORT_SHA[:12],flush=True)
    if not args.execute:
        print("EXP046_DRY_RUN_NO_MODEL_FORWARD",flush=True);return
    if args.out is None or args.out.exists() or args.out.parent.resolve()!=ROOT.resolve():
        raise RuntimeError("HOLD new report only in isolated knowledge worktree")
    idx,_,_=preflight()
    if check_shard_layout(*idx)!=1078285824:
        raise RuntimeError("HOLD model parameter count")
    from transformers import AutoConfig,AutoTokenizer,AutoModelForCausalLM
    tok=AutoTokenizer.from_pretrained(str(MID),local_files_only=True,trust_remote_code=False)
    sizes=[len(tok.encode(prompt(row),add_special_tokens=True)) for row in rows]
    if len(sizes)!=60 or not all(20<n<750 for n in sizes):
        raise RuntimeError("HOLD prompt-length inventory")
    print("EXP046_60_PROMPTS_TOKENIZED",min(sizes),max(sizes),flush=True)
    cfg=AutoConfig.from_pretrained(str(RL),trust_remote_code=True,local_files_only=True)
    if cfg.rope_parameters["rope_type"]!="yarn":
        raise RuntimeError("HOLD source rotary config")
    if available_gib()<36:
        raise RuntimeError("HOLD preload available RAM <36GiB")
    model=AutoModelForCausalLM.from_pretrained(str(MID),trust_remote_code=True,
        local_files_only=True,dtype=torch.bfloat16,use_safetensors=True,
        low_cpu_mem_usage=True)
    model.eval().to("cpu").requires_grad_(False)
    original=model.model.rotary_emb
    alternative=type(original)(cfg,device=torch.device("cpu"))
    if original.rope_type!="default" or alternative.rope_type!="yarn":
        raise RuntimeError("HOLD rotary modules")
    results={};conditioned={};lengths=[];beg=time.monotonic()
    try:
        for mode,rot in (("default",original),("yarn",alternative)):
            model.model.rotary_emb=rot
            scored=[]
            for j,row in enumerate(rows):
                val,nt=generate_one(model,tok,row)
                scored.append(val);lengths.append(nt)
                if (j+1)%6==0:
                    print("EXP046_PROGRESS",mode,j+1,"/",len(rows),
                          "elapsed",round(time.monotonic()-beg,1),flush=True)
            if len(scored)!=60:raise RuntimeError("HOLD incomplete mode")
            results[mode]=aggregate(rows,scored)
            conditioned[mode]=hit_conditioned(rows,scored,retrieval)
            for k in CONDITIONS:
                # Native EOS is a decoding outcome, not part of score().
                results[mode][k]["native_eos"]=sum(
                    int(x["native_eos"]) for row,x in zip(rows,scored) if row["kind"]==k)
                result=results[mode][k]
                if result["newline"]+result["native_eos"]+result["cap"]!=result["n"]:
                    raise RuntimeError("HOLD stopping reason counts")
            print("EXP046_MODE_COMPLETE",mode,
                  [(k,results[mode][k]["strict"]) for k in CONDITIONS],flush=True)
        if len(results)!=2 or len(lengths)!=120:
            raise RuntimeError("HOLD incomplete 120 model calls")
        output=report(results,lengths,time.monotonic()-beg,retrieval,conditioned)
        args.out.open("x",encoding="utf8").write(output)
        print("EXP046_COMPLETE",hashlib.sha256(output.encode()).hexdigest(),flush=True)
    finally:
        model.model.rotary_emb=original
        del model,alternative
        gc.collect()
if __name__=="__main__":main()
