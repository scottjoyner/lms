#!/usr/bin/env python3
"""EXP039B: source-disjoint, paired energy-equalized K2 gated-input ablations."""
import os
for k,v in {"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","CUDA_VISIBLE_DEVICES":"-1","OMP_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"1","TOKENIZERS_PARALLELISM":"false"}.items():os.environ[k]=v
import argparse, gc, hashlib, math
from pathlib import Path
import numpy as np
import torch
from k2_interpolation_pilot import MID,RL,preflight,check_shard_layout,available_gib,sha
from k2_causal_ablation import ROOT,LAYERS,CANDIDATES,collect_dev,prepare_eval,dev_cohort,dev_prepare,forward_cell,summarize,DEV_SHA,EVAL_SHA
PROTOCOL_SHA="d377a47a7446170610f17346df0a109a98e1256ec1660e5e14b5ed7c6f2e112a"
REPS=(0,1,2);TOL=.005
torch.set_num_threads(2)

def select(a,layer):
    if a.shape!=(24,5120) or not np.isfinite(a).all():raise RuntimeError("HOLD invalid dev activations")
    mag=np.abs(a.astype(np.float64)).mean(0)
    rank=np.lexsort((np.arange(5120),-mag))
    top=np.sort(rank[:32])
    controls=[]
    for r in REPS:
        choice=np.sort(np.random.default_rng(20261015+100*layer+r).choice(rank[32:256],size=32,replace=False))
        if set(top)&set(choice):raise RuntimeError("HOLD overlapping coordinates")
        controls.append(choice)
    return top,controls

def modify(x,top,random,group):
    if x.ndim!=3 or x.shape[0]!=1 or x.shape[-1]!=5120:raise RuntimeError("HOLD input topology")
    v=x[0,-1,:].float()
    tt=torch.as_tensor(top,dtype=torch.long)
    rr=torch.as_tensor(random,dtype=torch.long)
    et=float(v[tt].square().sum());er=float(v[rr].square().sum())
    total=float(v.square().sum())
    if not all(math.isfinite(q) and q>0 for q in (et,er,total)):raise RuntimeError("HOLD input energies")
    budget=min(et,er)
    indices=tt if group=="top" else rr
    src=et if group=="top" else er
    # EXP039B pre-registered quantization-aware scalar solver, not a
    # search over outcomes, neuron sets, questions or the error threshold.
    def realized_for(factor):
        q=(v[indices]*(1-factor)).to(dtype=x.dtype)
        actual=float((v[indices]-q.float()).square().sum())
        return q,actual
    best_factor=0.0
    best_val=None
    best_error=float("inf")
    lo,hi=0.0,1.0
    candidates=[0.0,1.0,math.sqrt(budget/src)]
    # Exactly 24 fixed bisection trials; no early outcome-based stopping.
    for _ in range(24):
        mid=(lo+hi)*0.5
        candidates.append(mid)
        _,actual=realized_for(mid)
        if actual<budget:lo=mid
        else:hi=mid
    for factor in candidates:
        candidate,actual=realized_for(factor)
        difference=abs(actual-budget)
        if (difference,factor)<(best_error,best_factor):
            best_error,best_factor,best_val=difference,factor,candidate
    y=x.clone()
    y[0,-1,indices]=best_val
    realized=float((v[indices]-y[0,-1,indices].float()).square().sum())
    error=abs(realized-budget)/budget
    if not math.isfinite(error) or error>TOL:raise RuntimeError("HOLD BF16 energy error "+str(error))
    if not torch.equal(x[:,:-1,:],y[:,:-1,:]):raise RuntimeError("HOLD altered previous tokens")
    return y,(budget,total,error,int(et<er))

def measure(model,items,ids,layer,top,rand,group):
    key=f"model.layers.{layer}.mlp.down_proj"
    mod=dict(model.named_modules()).get(key)
    if mod is None:raise RuntimeError("HOLD no hook site")
    receipts=[]
    def hook(module,args):
        y,r=modify(args[0],top,rand,group)
        receipts.append(r)
        return (y,)+tuple(args[1:])
    handler=mod.register_forward_pre_hook(hook)
    records=[]
    try:
        for item,unknown,gold,tokens in items:
            if available_gib()<22:raise RuntimeError("HOLD low available RAM")
            prev=len(receipts)
            with torch.inference_mode():
                z=model(**tokens,use_cache=False).logits[0,-1,ids].float().numpy().copy()
            if len(receipts)!=prev+1 or z.shape!=(3,) or not np.isfinite(z).all():
                raise RuntimeError("HOLD incomplete ablation")
            e=np.exp(z-z.max());p=e/e.sum()
            records.append((z,p,gold,unknown))
    finally:
        handler.remove()
    if len(records)!=10 or len(receipts)!=10:raise RuntimeError("HOLD non-ten result")
    return records,receipts

def pair_stats(t,r,baseline,tm,rm):
    wins=losses=ties=top_limited=0;diffs=[];fracs=[];errs=[]
    if not(len(t)==len(r)==len(baseline)==len(tm)==len(rm)==10):
        raise RuntimeError("HOLD sample counts")
    for (tz,tp,g,u),(rz,rp,rg,ru),(bz,bp,bg,bu),a,b in zip(t,r,baseline,tm,rm):
        if (g,u)!=(rg,ru) or (g,u)!=(bg,bu):raise RuntimeError("HOLD source permutation")
        if not math.isclose(a[0],b[0],rel_tol=1e-7,abs_tol=1e-8):
            raise RuntimeError("HOLD baseline activation energy mismatch")
        dt=float(np.abs(tp-bp).sum()*.5);dr=float(np.abs(rp-bp).sum()*.5)
        d=dt-dr
        wins+=int(d>1e-10);losses+=int(d< -1e-10);ties+=int(abs(d)<=1e-10)
        diffs.append(d);fracs.append(a[0]/a[1]);errs.extend((a[2],b[2]));top_limited+=a[3]
    return dict(delta_tv=float(np.mean(diffs)),wins=wins,losses=losses,ties=ties,
                budget_pct=float(np.mean(fracs)*100),max_error_pct=float(max(errs)*100),
                top_limited=top_limited)

def report(rows):
    lines=["# EXP-039 — Measured per-example energy-equalized gated-coordinate ablation","",
      "**2026-10-15, frozen K2 Mid2, research only.**",
      "Protocol SHA256: "+PROTOCOL_SHA+" (BF16 0.5% tolerance frozen before model forwards).",
      "Selection: 24 EXP036 prompts, topic receipt "+DEV_SHA+".",
      "Intervention evaluation: 10 EXP037 rotation0 questions, topic receipt "+EVAL_SHA+".",
      "Selection and evaluation article topics differ; however, prior research",
      "has already observed evaluation behavior and public SQuAD2 dev is not a",
      "training-independent private holdout.",
      "Three seeded rank-near random32 sets per layer, indices never persisted.",
      "At each last-token gated pre-down-projection input, same squared L2",
      "perturbation budget B=min(E_top,E_random) for target/control per pair.",
      "Actual BF16 realized energy checked within 0.5% on every hook.",
      "Exact no-op forward-hook/logit parity verified in both positional modes.",
      "",
      "## Paired intervention evidence",
      "",
      "| RoPE | Layer | Rep | Target TV | Random TV | Difference T-R | Target wins/10 | Random wins/10 | Ties | Target flips/10 | Random flips/10 | Target correct/10 | Random correct/10 | Budget as % of gated energy | Max BF16 energy error | Top limiting budget/10 |",
      "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for x in rows:
        p=x["paired"];t=x["top"];r=x["random"]
        lines.append(f"| {x['mode']} | {x['layer']} | {x['rep']} | "+
         f"{t['tv']:.7f} | {r['tv']:.7f} | {p['delta_tv']:+.7f} | "+
         f"{p['wins']} | {p['losses']} | {p['ties']} | {t['flips']} | {r['flips']} | "+
         f"{t['correct']} | {r['correct']} | {p['budget_pct']:.5f}% | "+
         f"{p['max_error_pct']:.4f}% | {p['top_limited']} |")
    lines+=["","## Interpretation and boundaries","",
      "- Matched perturbation magnitude is a stronger falsification control",
      "  than EXP038's mismatched full-zero comparison.",
      "- Equalized gated input energy does not equalize down_proj output",
      "  energy, learned Jacobian sensitivity, neuron correlation, or function.",
      "- The source was public dev data with only five evaluation article",
      "  groups; permutations and replicate sets are not independent source trials.",
      "- Individual golds, model traces, internal vectors and coordinate indices",
      "  remain transient and are NOT stored in the repository.",
      "- No semantic neuron specialization, direct Qwen teacher ancestry,",
      "  generalization or merged-checkpoint advantage established.",
      "- No saved model weights, routing, graph, NAS, credential, deployment,",
      "  release or arXiv submission.",""]
    return "\n".join(lines)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute",action="store_true")
    ap.add_argument("--out",type=Path)
    args=ap.parse_args()
    if sha(ROOT/"EXPERIMENT-039B-PROTOCOL.md")!=PROTOCOL_SHA:
        raise RuntimeError("HOLD frozen protocol SHA changed")
    index,_,_=preflight()
    if check_shard_layout(*index)!=1078285824:
        raise RuntimeError("HOLD pinned BF16 tensor count")
    print("EXP039B_PREFLIGHT_OK",PROTOCOL_SHA[:12],flush=True)
    if not args.execute:
        print("EXP039B_DRY_RUN_ONLY",flush=True)
        return
    if args.out is None or args.out.exists() or args.out.parent.resolve()!=ROOT.resolve():
        raise RuntimeError("HOLD report must be new isolated central knowledge Markdown")
    from transformers import AutoConfig,AutoTokenizer,AutoModelForCausalLM
    tokenizer=AutoTokenizer.from_pretrained(str(MID),local_files_only=True,trust_remote_code=False)
    dev=dev_prepare(dev_cohort(),tokenizer)
    evals=prepare_eval(tokenizer)
    ids=[]
    for label in CANDIDATES:
        t=tokenizer.encode(label,add_special_tokens=False)
        if len(t)!=1:raise RuntimeError("HOLD multi-token label")
        ids+=t
    cfg=AutoConfig.from_pretrained(str(RL),local_files_only=True,trust_remote_code=True)
    if cfg.rope_parameters["rope_type"]!="yarn":raise RuntimeError("HOLD YaRN source")
    if available_gib()<36:raise RuntimeError("HOLD pre-load RAM")
    model=AutoModelForCausalLM.from_pretrained(str(MID),local_files_only=True,
        trust_remote_code=True,dtype=torch.bfloat16,use_safetensors=True,low_cpu_mem_usage=True)
    model.eval().to("cpu").requires_grad_(False)
    back=model.model;native=back.rotary_emb
    alternative=type(native)(cfg,device=torch.device("cpu"))
    if native.rope_type!="default" or alternative.rope_type!="yarn":
        raise RuntimeError("HOLD rotary mismatch")
    try:
        vectors=collect_dev(model,dev)
        pairs={k:select(vectors[k],k) for k in LAYERS}
        del vectors
        gc.collect()
        rows=[]
        for mode,rotary in (("default",native),("yarn",alternative)):
            back.rotary_emb=rotary
            base=forward_cell(model,evals,ids,"baseline",{})
            idem=forward_cell(model,evals,ids,"identity",{})
            score=summarize(idem,base,"identity")
            print("EXP039B_IDENTITY_PASS",mode,score["correct"],flush=True)
            for layer in LAYERS:
                top,randoms=pairs[layer]
                for rep,random in enumerate(randoms):
                    t,tm=measure(model,evals,ids,layer,top,random,"top")
                    r,rm=measure(model,evals,ids,layer,top,random,"random")
                    entry={"mode":mode,"layer":layer,"rep":rep,
                       "top":summarize(t,base,"intervention"),
                       "random":summarize(r,base,"intervention"),
                       "paired":pair_stats(t,r,base,tm,rm)}
                    rows.append(entry)
                    print("EXP039B_PAIR",mode,layer,rep,
                          "T",round(entry["top"]["tv"],6),
                          "R",round(entry["random"]["tv"],6),
                          "W",entry["paired"]["wins"],
                          "max_err_pct",round(entry["paired"]["max_error_pct"],4),flush=True)
        if len(rows)!=12:raise RuntimeError("HOLD incomplete experiment")
        output=report(rows)
        args.out.open("x",encoding="utf8").write(output)
        print("EXP039B_COMPLETE",len(rows),"sha",hashlib.sha256(output.encode()).hexdigest(),flush=True)
    finally:
        back.rotary_emb=native
        del model,alternative
        gc.collect()
if __name__=="__main__":main()
