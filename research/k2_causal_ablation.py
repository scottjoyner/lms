#!/usr/bin/env python3
"""EXP038 — In-memory K2 last-token gated-coordinate causal zero-ablation.

Frozen Mid2 weights; 24 EXP036 dev selections; 10 EXP037 heldout intervention QAs.
No model artifact, raw prompt, neuron index inventory or per-item predictions saved.
"""
import os
for k,v in {"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","CUDA_VISIBLE_DEVICES":"-1",
            "TOKENIZERS_PARALLELISM":"false","OMP_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"1"}.items():
    os.environ[k]=v
import argparse, gc, hashlib, json
from pathlib import Path
import numpy as np
import torch
from k2_interpolation_pilot import MID, RL, preflight, check_shard_layout, available_gib, sha
from k2_factorial_disjoint import cohort as dev_cohort, prepare as dev_prepare, GROUP_SHA as DEV_SHA
from k2_label_rotation_falsification import cohort as evaluation_cohort, prompts as evaluation_prompts, GROUP_SHA as EVAL_SHA

ROOT=Path("/home/scott/git/wt-knowledge-k2-latex-20261014/50-Research/llm-research/latent-world-geometry-2026")
PROTOCOL_SHA="aecf8b3decb88c4cd3d218e62266e0aa8f76529b684cb319834ff80be945d714"
LAYERS=(7,14)
CANDIDATES=(" A"," B"," C")
CONDITIONS=("baseline","identity","layer7_top32","layer7_random32","layer14_top32","layer14_random32")
torch.set_num_threads(2)

def sha_text(x):
    return hashlib.sha256(x.encode()).hexdigest()

def prepare_eval(tok):
    cases=evaluation_cohort()
    # The cohort is article-disjoint from the dev cohort by verified source receipts.
    all_rotations=evaluation_prompts(cases,tok)
    if len(all_rotations)!=30:raise RuntimeError("HOLD missing EXP037 source items")
    one=[x for x in all_rotations if x[2]==0]
    if len(one)!=10 or len({x[0] for x in one})!=10:raise RuntimeError("HOLD incorrect eval items")
    return [(item,unknown,gold,toks) for item,unknown,rotation,gold,toks in one]

def choose_units(features,layer):
    if features.shape!=(24,5120) or not np.isfinite(features).all():
        raise RuntimeError("HOLD invalid dev feature shape/finiteness")
    mag=np.mean(np.abs(features.astype(np.float64)),axis=0)
    rank=np.lexsort((np.arange(5120,dtype=np.int64),-mag))
    top=rank[:32]
    rng=np.random.default_rng(20261015+layer)
    randomset=np.sort(rng.choice(rank[32:256],size=32,replace=False))
    if len(set(top)&set(randomset)) or len(set(top))!=32:
        raise RuntimeError("HOLD overlapping selection/control")
    energy=np.square(features.astype(np.float64))
    total=energy.sum()
    if total<=0:raise RuntimeError("HOLD zero baseline activation")
    frac=lambda indexes:float(energy[:,indexes].sum()/total)
    return top,randomset,{"top32_energy":frac(top),"random32_energy":frac(randomset),
                           "top32_mean_abs":float(mag[top].mean()),
                           "random32_mean_abs":float(mag[randomset].mean())}

def collect_dev(model,inputs):
    modules=dict(model.named_modules());hooks=[];current={};features={k:[] for k in LAYERS}
    def capture(k):
        def cb(module,args):
            x=args[0]
            if k in current or x.ndim!=3 or x.shape[0]!=1 or x.shape[-1]!=5120:
                raise RuntimeError("HOLD invalid development pre-hook")
            current[k]=x[0,-1,:].detach().float().cpu().numpy().copy()
        return cb
    for k in LAYERS:
        key=f"model.layers.{k}.mlp.down_proj"
        if key not in modules:raise RuntimeError("HOLD native layer missing")
        hooks.append(modules[key].register_forward_pre_hook(capture(k)))
    try:
        for article,unanswerable,tokens,gold in inputs:
            if available_gib()<22:raise RuntimeError("HOLD memory in development")
            current.clear()
            with torch.inference_mode():
                _=model(**tokens,use_cache=False).logits[0,-1,:3]
            if set(current)!=set(LAYERS):raise RuntimeError("HOLD incomplete development hooks")
            for k in LAYERS:features[k].append(current[k])
    finally:
        for handle in hooks:handle.remove()
    if len(features[7])!=24 or len(features[14])!=24:
        raise RuntimeError("HOLD development count mismatch")
    return {k:np.stack(features[k]) for k in LAYERS}

def hook_for(condition,selection):
    if condition=="baseline":return None,None
    if condition=="identity":
        def unchanged(module,args):
            return args
        return 7,unchanged
    spec={"layer7_top32":(7,"top"),"layer7_random32":(7,"random"),
          "layer14_top32":(14,"top"),"layer14_random32":(14,"random")}
    if condition not in spec:raise RuntimeError("HOLD unknown cell")
    layer,kind=spec[condition]
    indices=selection[layer][0 if kind=="top" else 1].tolist()
    if len(indices)!=32 or any(not 0<=i<5120 for i in indices):
        raise RuntimeError("HOLD intervention coordinates")
    def modify(module,args):
        x=args[0]
        if x.ndim!=3 or x.shape[0]!=1 or x.shape[-1]!=5120:
            raise RuntimeError("HOLD unexpected gated input to intervention")
        modified=x.clone()
        modified[0,-1,indices]=0.
        return (modified,)+tuple(args[1:])
    return layer,modify

def forward_cell(model,eval_records,ids,condition,selection):
    modules=dict(model.named_modules())
    layer,hook=hook_for(condition,selection)
    handle=None
    if layer is not None:
        key=f"model.layers.{layer}.mlp.down_proj"
        if key not in modules:raise RuntimeError("HOLD intervention module missing")
        handle=modules[key].register_forward_pre_hook(hook)
    row=[]
    try:
        for item,is_unanswerable,gold,tokens in eval_records:
            if available_gib()<22:raise RuntimeError("HOLD RAM during intervention")
            with torch.inference_mode():
                logits=model(**tokens,use_cache=False).logits[0,-1,ids].float().cpu().numpy().copy()
            if not np.isfinite(logits).all() or logits.shape!=(3,):
                raise RuntimeError("HOLD invalid conditional logits")
            exp=np.exp(logits-logits.max())
            probs=exp/exp.sum()
            if not np.isfinite(probs).all():raise RuntimeError("HOLD invalid probabilities")
            row.append((logits,probs,gold,is_unanswerable))
    finally:
        if handle is not None:handle.remove()
    if len(row)!=10:raise RuntimeError("HOLD incomplete cell")
    return row

def summarize(cell,baseline,condition):
    if len(cell)!=10 or len(baseline)!=10:raise RuntimeError("HOLD nonpaired cells")
    if condition=="identity":
        if any(not np.array_equal(a[0],b[0]) for a,b in zip(cell,baseline)):
            raise RuntimeError("HOLD identity control differs from no-hook baseline")
    correct=ans=unknown=flips=0
    nll=[];tvs=[];delta_gold=[];bch=[]
    for (logits,p,g,u),(blogits,bp,bg,bu) in zip(cell,baseline):
        if (g,u)!=(bg,bu):raise RuntimeError("HOLD item order mismatch")
        pred=int(p.argmax())
        correct+=int(pred==g)
        ans+=int(not u and pred==g)
        unknown+=int(u and pred==g)
        flips+=int(pred!=int(bp.argmax()))
        nll.append(-np.log(max(float(p[g]),1e-30)))
        tvs.append(float(np.abs(p-bp).sum()*0.5))
        delta_gold.append(float(p[g]-bp[g]))
        den=np.linalg.norm(p)*np.linalg.norm(bp)
        bch.append(float(np.dot(p,bp)/den))
    return {"correct":correct,"answerable":ans,"unknown":unknown,
            "nll":float(np.mean(nll)),"flips":flips,
            "tv":float(np.mean(tvs)),"delta_gold":float(np.mean(delta_gold)),
            "choice_cos":float(np.mean(bch))}

def build_report(dev,measures):
    lines=[
      "# EXP-038 — Real last-token gated-coordinate causal zero-ablation",
      "",
      "**2026-10-15, research-only, frozen Mid2 weights.** Model forward and input intervention really executed.",
      "Protocol SHA256: "+PROTOCOL_SHA,
      "Selection-only EXP-036 (24 input prompts) topic receipt: "+DEV_SHA,
      "Intervention-evaluation EXP-037 (10 questions, rotation=0) topic receipt: "+EVAL_SHA,
      "The evaluation article topics differ from coordinate-selection topics but EXP-037 behavioral outcomes were already observed previously.",
      "No independent model-training-unseen holdout is claimed. All tensors were in CPU memory only.",
      "Selected layer-7 and layer-14 top32 coordinates by DEV mean abs, compared with random32 among ranks33..256,",
      "seed 20261015+layer. Raw coordinate IDs, activations, passages and per-question results are not persisted.",
      "",
      "## Selection/control energy, measured only on development cohort",
      "",
      "| Native K2 layer | Top32 dev squared energy | Random32 dev squared energy | Top mean abs | Random mean abs |",
      "|---:|---:|---:|---:|---:|"]
    for layer in LAYERS:
        d=dev[layer][2]
        lines.append(f"| {layer} | {100*d['top32_energy']:.4f}% | {100*d['random32_energy']:.4f}% | {d['top32_mean_abs']:.6f} | {d['random32_mean_abs']:.6f} |")
    lines+=["","## Held-out intervention result (10 paired prompts, fixed source and position)",
       "",
       "| RoPE | Cell | Correct /10 | Answerable /5 | NOT STATED /5 | Choice NLL | Argmax changed /10 | TV vs baseline | Delta gold prob | Choice cosine vs baseline |",
       "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for mode in ("default","yarn"):
        for condition in CONDITIONS:
            m=measures[mode][condition]
            lines.append(f"| {mode} | {condition} | {m['correct']} | {m['answerable']} | {m['unknown']} | {m['nll']:.6f} | {m['flips']} | {m['tv']:.8f} | {m['delta_gold']:+.8f} | {m['choice_cos']:.8f} |")
    lines+=["","## Limitations and no-promotion gate",
      "",
      "Top32 is amplitude-selected, not question-specialized. Random32 is drawn from nearby high-ranked coordinates",
      "but amplitude/removed squared-energy may still differ; a larger effect does NOT identify semantic specialization.",
      "The forward-pre-hook zeroes only last-token 5120D gated MLP input before down_proj at exactly one layer.",
      "A changed probability is causal with respect to that *specific intervention* on the given model/input.",
      "No claim that one neuron causes semantic abstention, nor that model weights or Qwen ancestry are inferred.",
      "Ten evaluated questions from five article groups cannot provide reliable population-level inference.",
      "No per-item artifacts, persistent merged weights, training, production graph/NAS/routing/credential or deployments.",
      "Paper draft remains research-only; no merge, model promotion, or arXiv submission.",""]
    return "\n".join(lines)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute",action="store_true")
    ap.add_argument("--out",type=Path)
    args=ap.parse_args()
    if sha(ROOT/"EXPERIMENT-038-PROTOCOL.md")!=PROTOCOL_SHA:
        raise RuntimeError("HOLD frozen protocol SHA mismatch")
    indices,_,_=preflight()
    check_shard_layout(*indices)
    if DEV_SHA!="c1eac7118a917cc4505795767640e8321c671a346a41fa7bb9a22bf74e95577e":
        raise RuntimeError("HOLD unexpected development receipt")
    if EVAL_SHA!="fbf36a541ab0622ccca046390e9da24b0602393f501a2f40a68825e17f14045c":
        raise RuntimeError("HOLD unexpected evaluation receipt")
    print("EXP038_PREFLIGHT_OK dev24 eval10 proto",PROTOCOL_SHA[:12],flush=True)
    if not args.execute:
        print("DRY_RUN_NO_MODEL_FORWARD",flush=True)
        return
    if args.out is None or args.out.exists() or args.out.parent.resolve()!=ROOT.resolve():
        raise RuntimeError("HOLD output must be novel Markdown in isolated knowledge research worktree")
    from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(MID),local_files_only=True,trust_remote_code=False)
    dev=dev_prepare(dev_cohort(),tok)
    eval_records=prepare_eval(tok)
    choice=[]
    for txt in CANDIDATES:
        ids=tok.encode(txt,add_special_tokens=False)
        if len(ids)!=1:raise RuntimeError("HOLD option letter not single token")
        choice+=ids
    cfg=AutoConfig.from_pretrained(str(RL),local_files_only=True,trust_remote_code=True)
    if cfg.rope_parameters["rope_type"]!="yarn":raise RuntimeError("HOLD source for YaRN changed")
    if available_gib()<36:raise RuntimeError("HOLD pre-load resources")
    model=AutoModelForCausalLM.from_pretrained(str(MID),local_files_only=True,trust_remote_code=True,
        dtype=torch.bfloat16,use_safetensors=True,low_cpu_mem_usage=True)
    model.eval().to("cpu").requires_grad_(False)
    backbone=model.model
    original=backbone.rotary_emb
    if original.rope_type!="default":raise RuntimeError("HOLD Mid2 native RoPE mismatch")
    alternative=type(original)(cfg,device=torch.device("cpu"))
    if alternative.rope_type!="yarn":raise RuntimeError("HOLD YaRN mismatch")
    chosen={}
    try:
        dev_act=collect_dev(model,dev)
        for k in LAYERS:
            chosen[k]=choose_units(dev_act[k],k)
        del dev_act
        gc.collect()
        result={}
        for mode,rotary in (("default",original),("yarn",alternative)):
            backbone.rotary_emb=rotary
            cells={}
            baseline=forward_cell(model,eval_records,choice,"baseline",chosen)
            for condition in CONDITIONS:
                observed=baseline if condition=="baseline" else forward_cell(
                    model,eval_records,choice,condition,chosen)
                cells[condition]=summarize(observed,baseline,condition)
                m=cells[condition]
                print("EXP038_CELL",mode,condition,"acc",m["correct"],"flips",m["flips"],
                      "TV",round(m["tv"],8),flush=True)
            if len(cells)!=6:raise RuntimeError("HOLD mode incomplete")
            result[mode]=cells
        if len(result)!=2:raise RuntimeError("HOLD incomplete modes")
        report=build_report(chosen,result)
        args.out.open("x",encoding="utf-8").write(report)
        print("EXP038_PASS",sha_text(report),"rows",sum(len(x) for x in result.values()),flush=True)
    finally:
        backbone.rotary_emb=original
        del model,alternative
        gc.collect()
if __name__=="__main__":main()
