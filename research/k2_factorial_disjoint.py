#!/usr/bin/env python3
"""EXP036: real, CPU-only ephemeral K2 interpolation x RoPE factorial."""
import os
for k,v in {"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","CUDA_VISIBLE_DEVICES":"-1","TOKENIZERS_PARALLELISM":"false","OMP_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"1"}.items():os.environ[k]=v
import argparse,gc,hashlib,json,re,time
from pathlib import Path
import numpy as np
import torch
from k2_interpolation_pilot import MID,RL,ALPHAS,preflight,check_shard_layout,stage_weights,available_gib,sha
SRC=Path("/home/scott/git/.latent-squad2-dev-research/dev-v2.0.json")
SRC_SHA="80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8"
GROUP_SHA="c1eac7118a917cc4505795767640e8321c671a346a41fa7bb9a22bf74e95577e"
PROTOCOL_SHA="1935d03fb5e49790372435c0d9e9ff151228b32ac84fe33e5a01a38df6f4480c"
ROOT=Path("/home/scott/git/wt-knowledge-k2-latex-20261014/50-Research/llm-research/latent-world-geometry-2026")
LAYERS=(7,14)
torch.set_num_threads(2)
def h(text):return hashlib.sha256(text.encode()).hexdigest()

def cohort():
    if sha(SRC)!=SRC_SHA:raise RuntimeError("HOLD source hash")
    dataset=json.loads(SRC.read_text())
    if dataset["version"]!="v2.0":raise RuntimeError("HOLD dataset version")
    candidates=[]
    for sec in dataset["data"]:
        for para in sec["paragraphs"]:
            ctx=para["context"]
            if not 190<len(ctx)<1800:continue
            yes=[];no=[]
            for q in para["qas"]:
                if not q["question"].strip().lower().startswith(("who ","where ","which ")):continue
                if q["is_impossible"] and not q["answers"]:no.append(q)
                elif q["answers"] and all(re.fullmatch(r"[A-Za-z][A-Za-z\-]{1,25}",a["text"]) and ctx[a["answer_start"]:a["answer_start"]+len(a["text"])]==a["text"] for a in q["answers"]):yes.append(q)
            if yes and no:candidates.append((h(sec["title"]+ctx),sec["title"],ctx,yes,no))
    selected=[];seen=set()
    for digest,title,ctx,yes,no in sorted(candidates,key=lambda x:x[0]):
        if title in seen:continue
        seen.add(title)
        a=min(yes,key=lambda q:h(q["id"]));b=min(no,key=lambda q:h(q["id"]))
        selected.append((title,digest,a,b,ctx))
    if len(selected)<20:raise RuntimeError("HOLD insufficient distinct articles")
    chosen=selected[8:20]
    if set(x[0] for x in chosen)&set(x[0] for x in selected[:8]):raise RuntimeError("HOLD reused article")
    receipt=[(t,d,a["id"],b["id"],a["answers"][0]["text"]) for t,d,a,b,_ in chosen]
    if h(json.dumps(receipt,separators=(",",":")))!=GROUP_SHA:raise RuntimeError("HOLD cohort receipt mismatch")
    cases=[]
    for i,(title,digest,a,b,ctx) in enumerate(chosen):
        gold=a["answers"][0]["text"];foil=None
        for j in range(1,12):
            candidate=chosen[(i+j)%12][2]["answers"][0]["text"]
            if candidate.casefold()!=gold.casefold() and candidate.casefold() not in ctx.casefold():
                foil=candidate;break
        if foil is None:raise RuntimeError("HOLD no distractor")
        a_word,b_word=(gold,foil) if i%2==0 else (foil,gold)
        cases.append((i,False,ctx,a["question"],a_word,b_word,i%2))
        cases.append((i,True,ctx,b["question"],gold,foil,2))
    if len(cases)!=24 or sum(x[1] for x in cases)!=12:raise RuntimeError("HOLD cohort balance")
    return cases

def prepare(cases,tok):
    records=[]
    for i,unans,ctx,question,a,b,gold in cases:
        prompt=("Answer using only the passage. Choose A or B when supported, or C when not stated. Give one letter.\n"
                f"Passage: {ctx}\nQuestion: {question}\nA) {a}\nB) {b}\nC) NOT STATED\nAnswer:")
        tokens=tok(prompt,return_tensors="pt",add_special_tokens=True,truncation=False)
        tokens.pop("token_type_ids",None)
        if not 40<tokens["input_ids"].shape[1]<850:raise RuntimeError("HOLD prompt length")
        records.append((i,unans,tokens,gold))
    return records

def cka(a,b):
    x=np.asarray(a,dtype=np.float64);y=np.asarray(b,dtype=np.float64)
    if x.shape[0]!=24 or y.shape[0]!=24:raise RuntimeError("HOLD CKA sample count")
    k=x@x.T;l=y@y.T
    k=k-k.mean(0,keepdims=True)-k.mean(1,keepdims=True)+k.mean()
    l=l-l.mean(0,keepdims=True)-l.mean(1,keepdims=True)+l.mean()
    d=np.linalg.norm(k)*np.linalg.norm(l)
    if d<1e-18:raise RuntimeError("HOLD degenerate CKA")
    result=float((k*l).sum()/d)
    if result< -1e-6 or result>1.00001:raise RuntimeError("HOLD CKA bound")
    return result

def evaluate(model,prepared,choice_ids):
    modules=dict(model.named_modules());cache={};observed={k:[] for k in LAYERS}
    def make_hook(k):
        def hook(module,args):
            x=args[0]
            if k in cache or x.ndim!=3 or tuple(x.shape)[::2]!=(1,5120):
                raise RuntimeError("HOLD invalid MLP hook")
            cache[k]=x[0,-1,:].detach().float().numpy().copy()
        return hook
    handlers=[]
    for k in LAYERS:
        name=f"model.layers.{k}.mlp.down_proj"
        if name not in modules:raise RuntimeError("HOLD no hook layer")
        handlers.append(modules[name].register_forward_pre_hook(make_hook(k)))
    preds=[];probs=[];gold=[];unknown=[]
    try:
        for i,unans,toks,g in prepared:
            if available_gib()<22:raise RuntimeError("HOLD <22 GiB RAM")
            cache.clear()
            with torch.inference_mode():
                logits=model(**toks,use_cache=False).logits[0,-1,choice_ids].float()
                prob=torch.softmax(logits,dim=0).numpy()
            if set(cache)!=set(LAYERS) or not np.isfinite(prob).all():raise RuntimeError("HOLD incomplete hooks")
            for k in LAYERS:observed[k].append(cache[k])
            preds.append(int(np.argmax(prob)));probs.append(prob);gold.append(g);unknown.append(unans)
    finally:
        for h in handlers:h.remove()
    if len(preds)!=24:raise RuntimeError("HOLD partial model outputs")
    nll=float(np.mean([-np.log(max(float(p[g]),1e-30)) for p,g in zip(probs,gold)]))
    margin=float(np.mean([p[g]-max(p[j] for j in range(3) if j!=g) for p,g in zip(probs,gold)]))
    metrics={"total":sum(p==g for p,g in zip(preds,gold)),
       "answerable":sum(p==g for p,g,u in zip(preds,gold,unknown) if not u),
       "unknown":sum(p==g for p,g,u in zip(preds,gold,unknown) if u),
       "nll":round(nll,6),"margin":round(margin,6),
       "A":preds.count(0),"B":preds.count(1),"C":preds.count(2)}
    return metrics,{k:np.stack(observed[k]) for k in LAYERS}

def compare(activations,baseline,seed):
    out={}
    for k in LAYERS:
        x=activations[k].astype("float64");y=baseline[k].astype("float64")
        if x.shape!=y.shape or x.shape!=(24,5120):raise RuntimeError("HOLD hook shape")
        numerator=(x*y).sum(axis=1)
        denom=np.linalg.norm(x,axis=1)*np.linalg.norm(y,axis=1)
        cosine=float(np.mean(numerator/np.maximum(denom,1e-30)))
        mass=np.mean(np.abs(x),axis=0)
        perm=np.random.default_rng(seed+k)
        null=[]
        for _ in range(64):
            shuffled=perm.permutation(12)
            rows=[int(2*j+t) for j in shuffled for t in (0,1)]
            null.append(cka(x,y[rows]))
        out[k]={"abs":float(np.abs(x).mean()),"top32":float(np.sort(mass)[-32:].sum()/mass.sum()),
            "cos":cosine,"cka":cka(x,y),"null":float(np.mean(null))}
    return out

def stage(alpha,prepared,tok,rope_cfg,index):
    from transformers import AutoModelForCausalLM
    if available_gib()<36:raise RuntimeError("HOLD stage RAM")
    model=AutoModelForCausalLM.from_pretrained(str(MID),local_files_only=True,trust_remote_code=True,
           dtype=torch.bfloat16,use_safetensors=True,low_cpu_mem_usage=True)
    model.eval().to("cpu").requires_grad_(False)
    if stage_weights(model,index,alpha)!=255:raise RuntimeError("HOLD incomplete mixing")
    backbone=model.model;original=backbone.rotary_emb
    alternative=type(original)(rope_cfg,device=torch.device("cpu"))
    if original.rope_type!="default" or alternative.rope_type!="yarn":
        raise RuntimeError("HOLD rotary mismatch")
    ids=[]
    for choice in (" A"," B"," C"):
        tokids=tok.encode(choice,add_special_tokens=False)
        if len(tokids)!=1:raise RuntimeError("HOLD multichoice tokens")
        ids.extend(tokids)
    results={}
    try:
        for mode,rotary in (("default",original),("yarn",alternative)):
            backbone.rotary_emb=rotary
            results[mode]=evaluate(model,prepared,ids)
            m=results[mode][0]
            print("EXP036_CELL",alpha,mode,m["total"],m["answerable"],m["unknown"],m["nll"],flush=True)
    finally:
        backbone.rotary_emb=original
        del model,alternative
        gc.collect()
    return results

def render(rows):
    lines=["# EXP-036 — K2 interpolation × RoPE: real article-disjoint activation pilot","",
     "**2026-10-15 | Research-only. Article-disjoint from EXP029/033, public SQuAD2 dev, not independent model-training holdout.**",
     "Frozen protocol SHA256: "+PROTOCOL_SHA+"; selected 12-topic SHA256: "+GROUP_SHA+".",
     "Official SQuAD2 dev SHA256: "+SRC_SHA+". Source license: CC BY-SA 4.0, Rajpurkar et al.",
     "12 answerable and 12 impossible human-authored questions. Fixed A/B/C next-token choice, not generation.",
     "All 255 full-model tensors blended in memory; no checkpoint was written or deployed.",
     "Native model configurations are alpha=0/default and alpha=1/yarn; other cells control RoPE vs weights.","",
     "## Factorial behavior","",
     "| Alpha | RoPE | Correct/24 | Answerable/12 | NOT STATED/12 | NLL | Margin | A/B/C output |",
     "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for a,mode,metrics,summary in rows:
        lines.append(f"| {a:.2f} | {mode} | {metrics['total']} | {metrics['answerable']} | {metrics['unknown']} | {metrics['nll']:.6f} | {metrics['margin']:.6f} | {metrics['A']}/{metrics['B']}/{metrics['C']} |")
    lines+=["","## Gated MLP last-input-token observations vs alpha=0/default","",
       "Full 5120-coordinate layer-7 and layer-14 activations were transient only; no neuron IDs, vectors or text persisted.",
       "Cosine: matched K2 coordinates averaged over 24 prompts; CKA: centered 24x24 sample Gram geometry.",
       "Shuffle mean: 64 deterministic article-pair-preserving permutations (12 groups), descriptive only.","",
       "| Alpha | RoPE | Layer | Mean abs | Top32 mass | Cosine | CKA | Shuffle CKA mean |",
       "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for a,mode,metrics,summary in rows:
        for k,rec in summary.items():
            lines.append(f"| {a:.2f} | {mode} | {k} | {rec['abs']:.6f} | {rec['top32']:.6f} | {rec['cos']:.6f} | {rec['cka']:.6f} | {rec['null']:.6f} |")
    lines+=["","## Limitations","",
       "No statistical or causal neuron claim. Public SQuAD-dev samples may have been seen in training.",
       "Choice-position priors and foil selection confound semantic ability; same article contributes paired QAs.",
       "No free generation, source-independent heldout, calibration study, benchmark-wide superiority or release proof.",
       "No production model registration, routing, graph, NAS, credential, network or deployment changes.",""]
    return "\n".join(lines)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--execute",action="store_true")
    ap.add_argument("--out",type=Path)
    args=ap.parse_args()
    if sha(ROOT/"EXPERIMENT-036-PROTOCOL.md")!=PROTOCOL_SHA:raise RuntimeError("HOLD protocol SHA changed")
    indices,_,_=preflight()
    if check_shard_layout(*indices)!=1078285824:raise RuntimeError("HOLD tensor count")
    cases=cohort()
    print("EXP036_PREFLIGHT",len(cases),"items, 12 novel articles; pinned source",GROUP_SHA[:12],flush=True)
    if not args.execute:
        print("DRY_RUN_ONLY",flush=True);return
    if not args.out or args.out.exists() or args.out.parent.resolve()!=ROOT.resolve():
        raise RuntimeError("HOLD report path must be new and isolated")
    from transformers import AutoTokenizer,AutoConfig
    tok=AutoTokenizer.from_pretrained(str(MID),local_files_only=True,trust_remote_code=False)
    prepared=prepare(cases,tok)
    rope_cfg=AutoConfig.from_pretrained(str(RL),local_files_only=True,trust_remote_code=True)
    if rope_cfg.rope_parameters["rope_type"]!="yarn":raise RuntimeError("HOLD bad YaRN source")
    result={};start=time.monotonic()
    for a in ALPHAS:
        result[a]=stage(a,prepared,tok,rope_cfg,indices[1])
        print("EXP036_PROGRESS",a,round(time.monotonic()-start,1),flush=True)
    baseline=result[0.0]["default"][1]
    rows=[]
    for a in ALPHAS:
        for mode in ("default","yarn"):
            metrics,activations=result[a][mode]
            rows.append((a,mode,metrics,compare(activations,baseline,20261015)))
    report=render(rows)
    args.out.open("x",encoding="utf8").write(report)
    print("EXP036_COMPLETE",len(rows),"cells; report sha256",h(report),flush=True)
if __name__=="__main__":main()