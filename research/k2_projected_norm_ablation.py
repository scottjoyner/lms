#!/usr/bin/env python3
"""EXP040: K2 CPU gated-input interventions matched by down-projection norm."""
import os
for k,v in {"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","CUDA_VISIBLE_DEVICES":"-1","OMP_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"1","TOKENIZERS_PARALLELISM":"false"}.items():os.environ[k]=v
import argparse,gc,hashlib,math
import numpy as np
import torch
from k2_energy_equalized_ablation_b import (MID,RL,ROOT,LAYERS,CANDIDATES,
    DEV_SHA,EVAL_SHA,select,collect_dev,dev_prepare,dev_cohort,prepare_eval,
    preflight,check_shard_layout,available_gib,sha,forward_cell,summarize,pair_stats)
PROTOCOL_SHA="1cad05cb07fa9e35d38b42d181a082c5777e6a7b8d23dd29a96a395fe1067916"
TOL=.005
REPS=(0,1)
torch.set_num_threads(2)

def match_projected(x,W,top,random,group):
    """Match actual BF16-rounded delta * pinned W in float32, per original x."""
    if x.ndim!=3 or x.shape[0]!=1 or x.shape[-1]!=5120:
        raise RuntimeError("HOLD gated input shape")
    if W.ndim!=2 or tuple(W.shape)!=(1536,5120):
        raise RuntimeError("HOLD frozen projection shape")
    if x.dtype!=torch.bfloat16 or W.dtype!=torch.float32:
        raise RuntimeError("HOLD BF16 x / float32 W required")
    v=x[0,-1,:].float()
    ti=torch.as_tensor(top,dtype=torch.long)
    ri=torch.as_tensor(random,dtype=torch.long)
    if len(ti)!=32 or len(ri)!=32 or len(set(top)&set(random)):
        raise RuntimeError("HOLD control overlap/size")
    # W is immutable captured float32 copy of pinned BF16 matrix.
    WT=W[:,ti];WR=W[:,ri]
    et=float(torch.linalg.vector_norm(WT@v[ti]).square().item())
    er=float(torch.linalg.vector_norm(WR@v[ri]).square().item())
    full=float(torch.linalg.vector_norm(W@v).square().item())
    if not all(math.isfinite(z) and z>0 for z in (et,er,full)):
        raise RuntimeError("HOLD nonfinite or zero projection")
    budget=min(et,er)
    loc=ti if group=="top" else ri
    proj=WT if group=="top" else WR
    src=et if group=="top" else er
    def candidate(f):
        q=(v[loc]*(1-f)).to(torch.bfloat16)
        d=v[loc]-q.float()
        actual=float(torch.linalg.vector_norm(proj@d).square().item())
        return q,actual
    choices=[0.0,1.0,math.sqrt(budget/src)]
    lo,hi=0.0,1.0
    for _ in range(24):
        mid=(lo+hi)*.5
        choices.append(mid)
        _,energy=candidate(mid)
        if energy<budget:lo=mid
        else:hi=mid
    best=None
    for factor in choices:
        q,energy=candidate(factor)
        diff=abs(energy-budget)
        if best is None or (diff,factor)<(best[0],best[1]):
            best=(diff,factor,q)
    y=x.clone()
    y[0,-1,loc]=best[2]
    delta=v[loc]-y[0,-1,loc].float()
    energy=float(torch.linalg.vector_norm(proj@delta).square().item())
    error=abs(energy-budget)/budget
    if not math.isfinite(error) or error>TOL:
        raise RuntimeError("HOLD BF16 projected norm budget mismatch "+str(error))
    if not torch.equal(x[:,:-1,:],y[:,:-1,:]):
        raise RuntimeError("HOLD changed earlier tokens")
    # Correctly label this reference as float32 image norm of BF16 values.
    return y,(budget,full,error,int(et<er))

def measure(model,records,choice_ids,layer,top,random,group):
    name=f"model.layers.{layer}.mlp.down_proj"
    mod=dict(model.named_modules()).get(name)
    if mod is None or tuple(mod.weight.shape)!=(1536,5120):
        raise RuntimeError("HOLD missing real projection")
    if mod.weight.dtype!=torch.bfloat16:
        raise RuntimeError("HOLD original weights not BF16")
    W=mod.weight.detach().float().clone()
    hook_count=[];out=[]
    def hook(module,args):
        yy,receipt=match_projected(args[0],W,top,random,group)
        hook_count.append(receipt)
        return (yy,)+tuple(args[1:])
    handle=mod.register_forward_pre_hook(hook)
    try:
        for item,unknown,gold,tokens in records:
            if available_gib()<22:raise RuntimeError("HOLD memory while forward")
            n=len(hook_count)
            with torch.inference_mode():
                logits=model(**tokens,use_cache=False).logits[0,-1,choice_ids].float().numpy().copy()
            if len(hook_count)!=n+1 or logits.shape!=(3,) or not np.isfinite(logits).all():
                raise RuntimeError("HOLD hook count/nonfinite model output")
            z=np.exp(logits-logits.max());prob=z/z.sum()
            out.append((logits,prob,gold,unknown))
    finally:
        handle.remove()
    if len(out)!=10 or len(hook_count)!=10:
        raise RuntimeError("HOLD failed 10 item completion")
    return out,hook_count

def report(rows):
    lines=["# EXP-040 — Post-projection-norm-matched gated-coordinate ablations","",
      "**2026-10-15, real frozen K2 Mid2 CPU inference, research only.**",
      "Preregistered protocol SHA256: "+PROTOCOL_SHA,
      "Development cohort SHA256: "+DEV_SHA+" (24 EXP036 inputs).",
      "Evaluation cohort SHA256: "+EVAL_SHA+" (10 EXP037 rotation0 prompts).",
      "These public source topics are distinct from selection topics, but have",
      "been used in prior research and are NOT training-independent holdout.",
      "Actual frozen BF16 down_proj weights projected in float32 for matching.",
      "The objective is matched squared L2 norm of W.float32 @ BF16-realized",
      "selected last-token gated-input perturbations, NOT true BF16 GEMM",
      "output norm, input norm or downstream causal meaning.",
      "Per-example matching gate: <=0.500% projected norm-energy mismatch.",
      "Both native RoPE configurations passed exact no-op hook parity.",
      "",
      "## All preregistered paired comparisons","",
      "| RoPE | Layer | Rep | Target TV | Random TV | Delta TV T-R | Target > random /10 | Random > target /10 | Ties | Target flips/10 | Random flips/10 | Target correct/10 | Random correct/10 | Projection budget vs full proj % | Max projected energy error % | Target energy limiting /10 |",
      "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for e in rows:
        a=e["target"];b=e["random"];p=e["paired"]
        lines.append(f"| {e['rope']} | {e['layer']} | {e['rep']} | "+
            f"{a['tv']:.7f} | {b['tv']:.7f} | {p['delta_tv']:+.7f} | "+
            f"{p['wins']} | {p['losses']} | {p['ties']} | "+
            f"{a['flips']} | {b['flips']} | {a['correct']} | {b['correct']} | "+
            f"{p['budget_pct']:.5f}% | {p['max_error_pct']:.4f}% | "+
            f"{p['top_limited']} |")
    lines+=["","## Interpretation limitations","",
     "- Primary estimand is descriptive paired conditional-choice TV",
     "  difference under equalized *linear-image norm* of selected BF16",
     "  last-token gating coordinates (frozen down_proj weights).",
     "- All interventions have no gold-driven scalar optimization.",
     "- 10 questions (5 article topics) are not sufficient for stable",
     "  population inference, and were already observed in earlier experiments.",
     "- This does not equalize input energies, down_proj native GEMM roundoff,",
     "  residual Jacobians, source-grounded free-generation or semantic role.",
     "- No teacher ancestry, neuron specialization, merger benefit,",
     "  model promotion, deployment or arXiv publication.",""]
    return "\n".join(lines)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--out",type=__import__("pathlib").Path)
    args=parser.parse_args()
    if sha(ROOT/"EXPERIMENT-040-PROTOCOL.md")!=PROTOCOL_SHA:
        raise RuntimeError("HOLD protocol digest mismatch")
    idx,_,_=preflight()
    check_shard_layout(*idx)
    print("EXP040_PREFLIGHT_OK",PROTOCOL_SHA[:12],flush=True)
    if not args.execute:
        print("EXP040_DRY_RUN_ONLY",flush=True);return
    if args.out is None or args.out.exists() or args.out.parent.resolve()!=ROOT.resolve():
        raise RuntimeError("HOLD new isolated knowledge report required")
    from transformers import AutoConfig,AutoTokenizer,AutoModelForCausalLM
    tok=AutoTokenizer.from_pretrained(str(MID),local_files_only=True,trust_remote_code=False)
    dev=dev_prepare(dev_cohort(),tok)
    evals=prepare_eval(tok)
    ids=[]
    for lab in CANDIDATES:
        one=tok.encode(lab,add_special_tokens=False)
        if len(one)!=1:raise RuntimeError("HOLD non single choice token")
        ids.extend(one)
    cfg=AutoConfig.from_pretrained(str(RL),local_files_only=True,trust_remote_code=True)
    if cfg.rope_parameters["rope_type"]!="yarn":raise RuntimeError("HOLD not native YaRN")
    if available_gib()<36:raise RuntimeError("HOLD memory pre-load")
    model=AutoModelForCausalLM.from_pretrained(str(MID),local_files_only=True,
       trust_remote_code=True,dtype=torch.bfloat16,use_safetensors=True,low_cpu_mem_usage=True)
    model.eval().to("cpu").requires_grad_(False)
    old=model.model.rotary_emb
    yarn=type(old)(cfg,device=torch.device("cpu"))
    if old.rope_type!="default" or yarn.rope_type!="yarn":
        raise RuntimeError("HOLD rotary topology drift")
    try:
        activation=collect_dev(model,dev)
        selected={k:select(activation[k],k) for k in LAYERS}
        del activation;gc.collect()
        outputs=[]
        for mode,rotary in (("default",old),("yarn",yarn)):
            model.model.rotary_emb=rotary
            baseline=forward_cell(model,evals,ids,"baseline",{})
            identity=forward_cell(model,evals,ids,"identity",{})
            baseline_result=summarize(identity,baseline,"identity")
            print("EXP040_IDENTITY_PASS",mode,baseline_result["correct"],flush=True)
            for layer in LAYERS:
                top,controls=selected[layer]
                for rep in REPS:
                    rand=controls[rep]
                    t,tm=measure(model,evals,ids,layer,top,rand,"top")
                    r,rm=measure(model,evals,ids,layer,top,rand,"random")
                    row={"rope":mode,"layer":layer,"rep":rep,
                         "target":summarize(t,baseline,"intervention"),
                         "random":summarize(r,baseline,"intervention"),
                         "paired":pair_stats(t,r,baseline,tm,rm)}
                    outputs.append(row)
                    print("EXP040_PAIR",mode,layer,rep,
                          round(row["target"]["tv"],6),round(row["random"]["tv"],6),
                          "wins",row["paired"]["wins"],
                          "max_error",round(row["paired"]["max_error_pct"],5),flush=True)
        if len(outputs)!=8:raise RuntimeError("HOLD incomplete eight comparisons")
        text=report(outputs)
        args.out.open("x",encoding="utf8").write(text)
        print("EXP040_COMPLETE",len(outputs),"sha",hashlib.sha256(text.encode()).hexdigest(),flush=True)
    finally:
        model.model.rotary_emb=old
        del model,yarn
        gc.collect()
if __name__=="__main__":main()
