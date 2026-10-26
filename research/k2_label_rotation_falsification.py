#!/usr/bin/env python3
"""EXP037: five novel SQuAD topics, cyclic labels, K2 frozen CPU inference."""
import os
for k,v in {"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","CUDA_VISIBLE_DEVICES":"-1","TOKENIZERS_PARALLELISM":"false","OMP_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"1"}.items():os.environ[k]=v
import argparse,gc,hashlib,json,re
from pathlib import Path
import numpy as np
import torch
from k2_interpolation_pilot import MID,RL,preflight,check_shard_layout,stage_weights,available_gib,sha
ROOT=Path("/home/scott/git/wt-knowledge-k2-latex-20261014/50-Research/llm-research/latent-world-geometry-2026")
SOURCE=Path("/home/scott/git/.latent-squad2-dev-research/dev-v2.0.json")
SOURCE_SHA="80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8"
GROUP_SHA="fbf36a541ab0622ccca046390e9da24b0602393f501a2f40a68825e17f14045c"
PROTOCOL_SHA="95ef9965cee3f84338e46a136ac32430588d5e193e3053321b3daa15040d2877"
LAYERS=(7,14)
torch.set_num_threads(2)
def h(x):return hashlib.sha256(x.encode()).hexdigest()
def cohort():
    if sha(SOURCE)!=SOURCE_SHA:raise RuntimeError("HOLD source checksum")
    db=json.loads(SOURCE.read_text());eligible=[]
    if db["version"]!="v2.0":raise RuntimeError("HOLD source version")
    for section in db["data"]:
        for para in section["paragraphs"]:
            ctx=para["context"]
            if not 190<len(ctx)<1800:continue
            yes=[];no=[]
            for q in para["qas"]:
                if not q["question"].strip().lower().startswith(("who ","where ","which ")):continue
                if q["is_impossible"] and not q["answers"]:no.append(q)
                elif q["answers"] and all(re.fullmatch(r"[A-Za-z][A-Za-z\-]{1,25}",a["text"]) and ctx[a["answer_start"]:a["answer_start"]+len(a["text"])]==a["text"] for a in q["answers"]):yes.append(q)
            if yes and no:eligible.append((h(section["title"]+ctx),section["title"],ctx,yes,no))
    selected=[];seen=set()
    for d,t,ctx,yes,no in sorted(eligible,key=lambda x:x[0]):
        if t in seen:continue
        seen.add(t)
        a=min(yes,key=lambda q:h(q["id"]));b=min(no,key=lambda q:h(q["id"]))
        selected.append((t,d,a,b,ctx))
    if len(selected)!=25:raise RuntimeError("HOLD topic inventory")
    rest=selected[20:25]
    receipt=[(t,d,a["id"],b["id"],a["answers"][0]["text"]) for t,d,a,b,_ in rest]
    if h(json.dumps(receipt,separators=(",",":")))!=GROUP_SHA:raise RuntimeError("HOLD cohort receipt")
    items=[]
    for i,(t,d,a,b,ctx) in enumerate(rest):
        gold=a["answers"][0]["text"];foil=None
        for j in range(1,5):
            c=rest[(i+j)%5][2]["answers"][0]["text"]
            if c.casefold()!=gold.casefold() and c.casefold() not in ctx.casefold():
                foil=c;break
        if foil is None:raise RuntimeError("HOLD no foil")
        items.append((i,False,ctx,a["question"],(gold,foil,"NOT STATED"),0))
        items.append((i,True,ctx,b["question"],(gold,foil,"NOT STATED"),2))
    if len(items)!=10 or sum(x[1] for x in items)!=5:raise RuntimeError("HOLD wrong cohort")
    return items

def prompts(items,tok):
    out=[]
    for item,(article,unans,ctx,q,semantic,gold) in enumerate(items):
        for rotation in range(3):
            options=[semantic[(j-rotation)%3] for j in range(3)]
            text=("Answer based ONLY on the passage. Select A/B/C for the supported answer, "
                 "or the option NOT STATED if the passage does not establish an answer. "
                 "Reply with one letter.\nPassage: "+ctx+"\nQuestion: "+q+"\n"+
                 "\n".join(f"{j}) {v}" for j,v in zip("ABC",options))+"\nAnswer:")
            inputs=tok(text,return_tensors="pt",truncation=False,add_special_tokens=True)
            inputs.pop("token_type_ids",None)
            if not 30<inputs["input_ids"].shape[1]<850:raise RuntimeError("HOLD long prompt")
            out.append((item,unans,rotation,(gold+rotation)%3,inputs))
    if len(out)!=30:raise RuntimeError("HOLD incomplete rotation")
    return out

def evaluate(model,inputs,choice_ids):
    mods=dict(model.named_modules());cache={};hooks={k:[] for k in LAYERS};handles=[]
    def hook(k):
        def cb(mod,args):
            x=args[0]
            if k in cache or x.ndim!=3 or x.shape[0]!=1 or x.shape[-1]!=5120:
                raise RuntimeError("HOLD incorrect hook")
            cache[k]=x[0,-1,:].detach().float().numpy().copy()
        return cb
    for k in LAYERS:
        name=f"model.layers.{k}.mlp.down_proj"
        if name not in mods:raise RuntimeError("HOLD missing MLP module")
        handles.append(mods[name].register_forward_pre_hook(hook(k)))
    preds=[];correct=[];unknown=[];semantic=[];nll=[]
    try:
        for item,impossible,rotation,gold,toks in inputs:
            if available_gib()<22:raise RuntimeError("HOLD resource admission")
            cache.clear()
            with torch.inference_mode():
                z=model(**toks,use_cache=False).logits[0,-1,choice_ids].float().numpy()
            if set(cache)!=set(LAYERS) or not np.isfinite(z).all():raise RuntimeError("HOLD incomplete forward")
            for k in LAYERS:hooks[k].append(cache[k])
            a=np.exp(z-np.max(z));p=a/a.sum();pred=int(np.argmax(p))
            preds.append(pred);correct.append(pred==gold);unknown.append(impossible)
            semantic.append((pred-rotation)%3)
            nll.append(-float(np.log(max(p[gold],1e-30))))
    finally:
        for hndl in handles:hndl.remove()
    if len(correct)!=30:raise RuntimeError("HOLD missing questions")
    out={"total":sum(correct),"answerable":sum(c for c,u in zip(correct,unknown) if not u),
      "unknown":sum(c for c,u in zip(correct,unknown) if u),
      "all3":sum(all(correct[i:i+3]) for i in range(0,30,3)),
      "ans_all3":sum(all(correct[i:i+3]) for i in range(0,30,3) if not unknown[i]),
      "unk_all3":sum(all(correct[i:i+3]) for i in range(0,30,3) if unknown[i]),
      "semantic_stable":sum(len(set(semantic[i:i+3]))==1 for i in range(0,30,3)),
      "rot_correct":"/".join(str(sum(correct[j] for j in range(r,30,3))) for r in range(3)),
      "pred_labels":"/".join(str(preds.count(j)) for j in range(3)),
      "nll":round(float(np.mean(nll)),6)}
    geometry={}
    for k in LAYERS:
        x=np.stack(hooks[k]).reshape(10,3,5120).astype(np.float64)
        for rot in (1,2):
            num=np.sum(x[:,0,:]*x[:,rot,:],axis=1)
            den=np.linalg.norm(x[:,0,:],axis=1)*np.linalg.norm(x[:,rot,:],axis=1)
            geometry[(k,rot)]=round(float(np.mean(num/np.maximum(den,1e-30))),6)
    return out,geometry

def stage(alpha,prepared,model_rope,index,ids):
    from transformers import AutoModelForCausalLM
    if available_gib()<36:raise RuntimeError("HOLD pre-stage RAM")
    model=AutoModelForCausalLM.from_pretrained(str(MID),trust_remote_code=True,
         local_files_only=True,dtype=torch.bfloat16,use_safetensors=True,low_cpu_mem_usage=True)
    model.eval().to("cpu").requires_grad_(False)
    stage_weights(model,index,alpha)
    original=model.model.rotary_emb
    ya=type(original)(model_rope,device=torch.device("cpu"))
    if original.rope_type!="default" or ya.rope_type!="yarn":raise RuntimeError("HOLD invalid RoPE")
    out={}
    try:
        for mode,rot in (("default",original),("yarn",ya)):
            model.model.rotary_emb=rot
            out[mode]=evaluate(model,prepared,ids)
            m=out[mode][0]
            print("EXP037_CELL",alpha,mode,m["total"],m["answerable"],m["unknown"],m["all3"],m["nll"],flush=True)
    finally:
        model.model.rotary_emb=original
        del model,ya
        gc.collect()
    return out

def report(rows):
    out=["# EXP-037 — Three-way label placement falsification",
      "","2026-10-15, measured frozen CPU forwards; research-only.",
      "Protocol SHA256: "+PROTOCOL_SHA+"; novel 5-topic cohort SHA256: "+GROUP_SHA,
      "Public SQuAD 2 dev SHA256: "+SOURCE_SHA+"; CC BY-SA 4.0.",
      "Ten human-annotated questions (5 answerable, 5 impossible), each across all A/B/C semantic-label rotations.",
      "Article-topic groups are disjoint from EXP029, EXP033 and EXP036, not model-training-unseen.",
      "30 correlated prompts per model cell; four cells; no checkpoints saved.","",
      "## Accuracy and label robustness","",
      "| Weights alpha | RoPE | Correct /30 | Answerable /15 | NOT STATED /15 | All three correct /10 | Answerable robust /5 | NOT STATED robust /5 | Semantic choice stable /10 | Correct by rotation /10 each | Predicted A/B/C | Choice NLL |",
      "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|"]
    for alpha in (0.,1.):
        for mode in ("default","yarn"):
            m,g=rows[alpha][mode]
            out.append(f"| {alpha:.0f} | {mode} | {m['total']} | {m['answerable']} | {m['unknown']} | {m['all3']} | {m['ans_all3']} | {m['unk_all3']} | {m['semantic_stable']} | {m['rot_correct']} | {m['pred_labels']} | {m['nll']:.6f} |")
    out+=["","## Hook sensitivity to label rotation","",
          "Mean last-input-token gated MLP 5120D vector cosine across same question, rotation0 vs rotation1/2.",
          "No neuron ID, per-prompt activation or text exported; observational only.","",
          "| Alpha | RoPE | Layer | 0 vs 1 cosine | 0 vs 2 cosine |",
          "|---:|---|---:|---:|---:|"]
    for alpha in (0.,1.):
        for mode in ("default","yarn"):
            m,g=rows[alpha][mode]
            for k in LAYERS:out.append(f"| {alpha:.0f} | {mode} | {k} | {g[(k,1)]:.6f} | {g[(k,2)]:.6f} |")
    out+=["","## Boundaries",
      "Cyclic permutations are repeated questions, not 30 independent ground-truth items.",
      "Five articles cannot establish population significance; foil choices and instruction effects remain confounders.",
      "Strong A/B/C preference invalidates direct semantic improvement assertions.",
      "No merge, deployment, graph, NAS, credential, routing or production changes.",""]
    return "\n".join(out)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--execute",action="store_true");ap.add_argument("--out",type=Path)
    args=ap.parse_args()
    if sha(ROOT/"EXPERIMENT-037-PROTOCOL.md")!=PROTOCOL_SHA:raise RuntimeError("HOLD protocol SHA")
    index,_,_=preflight();check_shard_layout(*index)
    items=cohort()
    print("EXP037_PREFLIGHT",len(items),"novel questions",GROUP_SHA[:12],flush=True)
    if not args.execute:
        print("DRY_RUN_ONLY",flush=True);return
    if not args.out or args.out.exists() or args.out.parent.resolve()!=ROOT.resolve():
        raise RuntimeError("HOLD new research report required")
    from transformers import AutoTokenizer,AutoConfig
    tok=AutoTokenizer.from_pretrained(str(MID),trust_remote_code=False,local_files_only=True)
    prepared=prompts(items,tok)
    cfg=AutoConfig.from_pretrained(str(RL),trust_remote_code=True,local_files_only=True)
    if cfg.rope_parameters["rope_type"]!="yarn":raise RuntimeError("HOLD wrong YaRN config")
    ids=[]
    for t in (" A"," B"," C"):
        j=tok.encode(t,add_special_tokens=False)
        if len(j)!=1:raise RuntimeError("HOLD not single token")
        ids+=j
    results={a:stage(a,prepared,cfg,index[1],ids) for a in (0.,1.)}
    output=report(results)
    args.out.open("x",encoding="utf-8").write(output)
    print("EXP037_PASS",hashlib.sha256(output.encode()).hexdigest(),flush=True)
if __name__=="__main__":main()