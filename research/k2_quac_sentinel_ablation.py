#!/usr/bin/env python3
"""EXP044: pre-registered repeated original-vs-stripped QuAC sentinel K2 study."""
import os
for k,v in {"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","CUDA_VISIBLE_DEVICES":"-1",
   "TOKENIZERS_PARALLELISM":"false","OMP_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"1",
   "MKL_NUM_THREADS":"2"}.items():os.environ[k]=v
import argparse,gc,hashlib,time
from pathlib import Path
import torch
from k2_interpolation_pilot import (MID,RL,TOK_SHA,IDX_SHA,CONFIG_SHA,
    preflight,check_shard_layout,sha,available_gib)
from k2_quac_abstention import generate_one
from k2_quac_sentinel_counterfactual import (pairs,pair_digest)
import k2_quac_independent_fixture as q

ROOT=Path("/home/scott/git/wt-knowledge-k2-latex-20261014/50-Research/llm-research/latent-world-geometry-2026")
PROTOCOL_SHA="0e1db73619c7bf594a23ec75d94c7d543551f0dea74fffe267a3f8c489f39539"
PAIRED_SHA="b4c9246ea38355a82772d3e03ae1d73ca3edada1d2e188845cff5d04e29cd83f"
CLEAN_SHA="af911d3ed8cd3c347776e06452490f45ff6b65bb9fa967af806096c9b1b7ef3b"
ORIGINAL_RESULT_SHA="9a99d5b0b512daacb4c19b6c5edbd731e13b8048a0b32e8ebc025d19828945ff"
KINDS=("answerable","unanswerable","withheld","swapped")
MODES=("default","yarn")
ARMS=("original","no_sentinel")
torch.set_num_threads(2)

def frozen_preflight():
    if sha(ROOT/"EXPERIMENT-044-PROTOCOL-2026-10-15.md")!=PROTOCOL_SHA:
        raise RuntimeError("HOLD frozen 044 protocol changed")
    if sha(ROOT/"EXPERIMENT-043-RESULTS-2026-10-15.md")!=ORIGINAL_RESULT_SHA:
        raise RuntimeError("HOLD archived 043 source report changed")
    for i,root in enumerate((MID,RL)):
        if not root.is_dir() or root.is_symlink():
            raise RuntimeError("HOLD missing/symlink pinned K2")
        for file,expected in (("model.safetensors.index.json",IDX_SHA[i]),
                              ("config.json",CONFIG_SHA[i]),
                              ("tokenizer.json",TOK_SHA)):
            if sha(root/file)!=expected:
                raise RuntimeError("HOLD pinned checkpoint/tokenizer sha")
    original,cleaned=pairs()
    if pair_digest()!=PAIRED_SHA or q.receipt(cleaned)!=CLEAN_SHA:
        raise RuntimeError("HOLD cleaned paired prompt checksum")
    return original,cleaned

def old_table():
    text=(ROOT/"EXPERIMENT-043-RESULTS-2026-10-15.md").read_text()
    body=text.split("## Complete strict-format and abstention outcomes",1)[1].split("## Boundaries",1)[0]
    out={}
    for line in body.splitlines():
        if not(line.startswith("| default |") or line.startswith("| yarn |")):
            continue
        v=[x.strip() for x in line.split("|")[1:-1]]
        if len(v)!=12:raise RuntimeError("HOLD archival QuAC row width")
        mode,kind=v[:2]
        out[(mode,kind)]={"n":int(v[2]),"strict":int(v[3]),"first_line":int(v[4]),
          "mean_f1":float(v[5]),"contains":int(v[6]),"unknown":int(v[7]),
          "newline":int(v[8]),"native_eos":int(v[9]),"cap":int(v[10]),"empty":int(v[11])}
    if len(out)!=8:raise RuntimeError("HOLD archived QuAC missing rows")
    return out

def summarize_pairs(original,cleaned,source,outputs):
    if len(original)!=36 or len(cleaned)!=36 or len(outputs)!=2:
        raise RuntimeError("HOLD incomplete paired source strata")
    computed={}
    for mode in MODES:
        computed[mode]={}
        if any(arm not in outputs[mode] or len(outputs[mode][arm])!=36 for arm in ARMS):
            raise RuntimeError("HOLD incomplete paired K2 original/clean output")
        for arm in ARMS:
            rows=original if arm=="original" else cleaned
            scores=outputs[mode][arm]
            stats=q.aggregate(rows,scores)
            for kind in KINDS:
                stats[kind]["native_eos"]=sum(x["native_eos"] for r,x in zip(rows,scores)
                        if r["kind"]==kind)
                item=stats[kind]
                if item["newline"]+item["native_eos"]+item["cap"]!=item["n"]:
                    raise RuntimeError("HOLD incompletely classified decoder")
            computed[mode][arm]=stats
    # Reproduce the original 72 model generations by category (strict and
    # all auxiliary counts), so a rerun with changed runtime cannot be
    # silently interpreted as a source-format intervention.
    old=old_table()
    for mode in MODES:
        for kind in KINDS:
            observed=computed[mode]["original"][kind]
            archived=old[(mode,kind)]
            for key,value in archived.items():
                if key=="mean_f1":
                    if abs(observed[key]-value)>0.000001:
                        raise RuntimeError("HOLD original arm drift: mean token F1")
                elif observed[key]!=value:
                    raise RuntimeError("HOLD original arm drift: "+mode+"/"+kind+"/"+key)
    paired={}
    for mode in MODES:
        paired[mode]={}
        first=outputs[mode]["original"];other=outputs[mode]["no_sentinel"]
        for kind in KINDS:
            subset=[(a,b) for r,a,b in zip(original,first,other) if r["kind"]==kind]
            if len(subset)!=computed[mode]["original"][kind]["n"]:
                raise RuntimeError("HOLD source pairing mismatch")
            paired[mode][kind]={
              "strict_gained":sum(int(not x["strict"] and y["strict"]) for x,y in subset),
              "strict_lost":sum(int(x["strict"] and not y["strict"]) for x,y in subset),
              "unknown_gained":sum(int(not x["unknown"] and y["unknown"]) for x,y in subset),
              "unknown_lost":sum(int(x["unknown"] and not y["unknown"]) for x,y in subset)}
            if kind=="withheld" and any(x!=y for x,y in subset):
                raise RuntimeError("HOLD unchanged withheld control not bitwise stable")
    return computed,paired

def report(totals,paired,lengths,seconds):
    lines=["# EXP-044 — QuAC terminal CANNOTANSWER marker ablation",
      "",
      "**2026-10-15 | Complete paired frozen Mid2 K2 CPU generation (research only).**",
      "",
      "## Locked input/model provenance",
      "",
      "- Exact frozen protocol SHA256: "+PROTOCOL_SHA,
      "- Original QuAC EXP043 aggregate source SHA256: "+ORIGINAL_RESULT_SHA,
      "- Original-cleaned source/gold/question paired SHA256: "+PAIRED_SHA,
      "- Cleaned prompt/gold SHA256: "+CLEAN_SHA,
      "- Official QuAC source SHA256: "+q.SOURCE_SHA,
      "- Exactly 30/36 original contexts had a unique terminal dataset sentinel,",
      "  CANNOTANSWER; only that terminal marker and trailing whitespace were removed.",
      "- Exactly six withheld sources were unchanged in both source arms.",
      "- No article, question, gold label or dialog history changed.",
      "- 144 CPU greedy max16 generation calls (2 RoPE x 2 arms x 36 cases),",
      "  same EXP042 first-generated-newline stop, pinned Mid2 learned weights.",
      "- All 72 original-repeat generations matched archived EXP043 aggregate",
      "  strict, UNKNOWN, F1 and stopping metrics exactly.",
      "- QuAC original question labels are human-annotated. Withheld and",
      "  swapped contexts remain researcher-imposed UNKNOWN controls.",
      "- Public QuAC content is not model-training-independent.",
      "",
      "## Complete 16-cell strict and abstention outcome matrix",
      "",
      "| RoPE | Source arm | Case | N | Strict full | First line | Mean token F1 | Gold contained | Exact UNKNOWN | Newline | Native EOS | Cap | Empty |",
      "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for mode in MODES:
        for arm in ARMS:
            for kind in KINDS:
                s=totals[mode][arm][kind]
                lines.append(f"| {mode} | {arm} | {kind} | {s['n']} | "
                  f"{s['strict']} | {s['first_line']} | {s['mean_f1']:.6f} | "
                  f"{s['contains']} | {s['unknown']} | {s['newline']} | "
                  f"{s['native_eos']} | {s['cap']} | {s['empty']} |")
    lines+=["",
      "## Paired per-question changes (all prompts included)",
      "",
      "| RoPE | Kind | N | Strict gained | Strict lost | UNKNOWN gained | UNKNOWN lost |",
      "|---|---|---:|---:|---:|---:|---:|"]
    for mode in MODES:
        for kind in KINDS:
            p=paired[mode][kind]
            n=totals[mode]["original"][kind]["n"]
            lines.append(f"| {mode} | {kind} | {n} | {p['strict_gained']} | "
               f"{p['strict_lost']} | {p['unknown_gained']} | {p['unknown_lost']} |")
    lines+=["","## Interpretive boundaries","",
      "- This is a repeated public 12-dialog source-format counterfactual,",
      "  NOT independent model training holdout nor official QuAC benchmarking.",
      "- Removing a terminal text marker may change prompt completion priors,",
      "  model attention and context tokenization, not only semantic evidence.",
      "- The 36 repeated cases are paired within mode and share 12 dialogs;",
      "  inferential significance, teacher ancestry and semantic causality are",
      "  not established by this experiment alone.",
      "- No raw output, question text, article title, QID, logits, per-item",
      "  model responses or activations were archived.",
      f"- Prompt lengths min/max: {min(lengths)} / {max(lengths)}.",
      f"- Full repeated-and-cleaned inference duration: {seconds:.2f} seconds.",
      "- No model checkpoint writes, training, serving, graph/NAS, credential,",
      "  routing, network, deployment or arXiv submission.",
      ""]
    return "\n".join(lines)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute",action="store_true")
    ap.add_argument("--out",type=Path)
    args=ap.parse_args()
    old,clean=frozen_preflight()
    print("EXP044_PREFLIGHT_OK",len(old),PAIRED_SHA[:12],flush=True)
    if not args.execute:
        print("EXP044_MODEL_FREE_DRY_RUN",flush=True);return
    if args.out is None or args.out.exists() or args.out.parent.resolve()!=ROOT.resolve():
        raise RuntimeError("HOLD new isolated report location only")
    idx,_,_=preflight()
    if check_shard_layout(*idx)!=1078285824:
        raise RuntimeError("HOLD unexpected checkpoint size")
    from transformers import AutoModelForCausalLM,AutoConfig,AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(MID),trust_remote_code=False,local_files_only=True)
    lens=[len(tok.encode(q.prompt(row),add_special_tokens=True))
          for arm in (old,clean) for row in arm]
    if len(lens)!=72 or not all(20<n<750 for n in lens):
        raise RuntimeError("HOLD invalid source prompt lengths")
    print("EXP044_PROMPTS_TOKENIZED",min(lens),max(lens),flush=True)
    cfg=AutoConfig.from_pretrained(str(RL),trust_remote_code=True,local_files_only=True)
    if cfg.rope_parameters["rope_type"]!="yarn":
        raise RuntimeError("HOLD YaRN config drift")
    if available_gib()<36:raise RuntimeError("HOLD RAM <36GiB before model load")
    model=AutoModelForCausalLM.from_pretrained(str(MID),trust_remote_code=True,
      local_files_only=True,dtype=torch.bfloat16,use_safetensors=True,low_cpu_mem_usage=True)
    model.eval().to("cpu").requires_grad_(False)
    original_rotary=model.model.rotary_emb
    other_rotary=type(original_rotary)(cfg,device=torch.device("cpu"))
    if original_rotary.rope_type!="default" or other_rotary.rope_type!="yarn":
        raise RuntimeError("HOLD positional module topology")
    outputs={};sizes=[];start=time.monotonic()
    try:
        for mode,rot in (("default",original_rotary),("yarn",other_rotary)):
            model.model.rotary_emb=rot
            outputs[mode]={}
            for arm,rows in (("original",old),("no_sentinel",clean)):
                records=[]
                for j,row in enumerate(rows):
                    item,n=generate_one(model,tok,row)
                    records.append(item);sizes.append(n)
                    if (j+1)%6==0:
                        print("EXP044_PROGRESS",mode,arm,j+1,"/",len(rows),
                          "elapsed",round(time.monotonic()-start,1),flush=True)
                if len(records)!=36:
                    raise RuntimeError("HOLD incomplete run arm")
                outputs[mode][arm]=records
        if len(sizes)!=144:raise RuntimeError("HOLD incomplete 144 K2 calls")
        totals,pair_delta=summarize_pairs(old,clean,q.source(),outputs)
        out=report(totals,pair_delta,sizes,time.monotonic()-start)
        args.out.open("x",encoding="utf8").write(out)
        print("EXP044_COMPLETE",hashlib.sha256(out.encode()).hexdigest(),flush=True)
    finally:
        model.model.rotary_emb=original_rotary
        del model,other_rotary
        gc.collect()

if __name__=="__main__":main()
