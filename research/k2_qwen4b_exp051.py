#!/usr/bin/env python3
"""EXP051 preregistered synthetic text-only counterfactual K2/Qwen pilot."""
import os
for k,v in {"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","CUDA_VISIBLE_DEVICES":"-1",
            "TOKENIZERS_PARALLELISM":"false","OMP_NUM_THREADS":"4","MKL_NUM_THREADS":"4"}.items():
    os.environ[k]=v
import argparse, collections, gc, hashlib, json, statistics, time
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parent
PROTOCOL_SHA="63c3c50afc2f98bb2d5928da180e7dc1f79f52b7672378ec829fb6f7209c4fdf"
MODELS={
 "k2":Path("/home/scott/models/research/K2-Horizon-3.7B"),
 "qwen":Path("/home/scott/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B-Base/snapshots/1001bb4d826a52d1f399e183466143f4da7b741b")}
TOPICS=(
 ("bronze marker","quartz marker","Rilo","Sema","violet marker"),
 ("jade ticket","pearl ticket","Anvi","Bexo","amber ticket"),
 ("cobalt cipher","silver cipher","Nera","Tovi","violet cipher"),
 ("scarlet badge","ivory badge","Elra","Pavo","teal badge"),
 ("cedar disk","birch disk","Mira","Leno","maple disk"),
 ("indigo card","plum card","Ivo","Dava","lime card"),
 ("north relay","south relay","Rexa","Yuni","west relay"),
 ("opal parcel","onyx parcel","Tara","Zeno","ruby parcel"),
)
ARMS=("supported_original","supported_swapped","unrelated_original",
      "unrelated_swapped","supported_withheld")
torch.set_num_threads(4)
def sha_file(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()

def available_gib():
    line=next(x for x in Path("/proc/meminfo").read_text().splitlines()
              if x.startswith("MemAvailable:"))
    return int(line.split()[1])/1024**2

def make():
    out=[]
    for i,(a,b,x,y,unrelated) in enumerate(TOPICS):
        if len({a,b,unrelated})!=3 or x==y:raise RuntimeError("HOLD invalid topic")
        for arm in ARMS:
            src=(f"The {a} was assigned to {x}. The {b} was assigned to {y}."
                 if arm.endswith("original") else
                 f"The {a} was assigned to {y}. The {b} was assigned to {x}."
                 if arm.endswith("swapped") else
                 "The register contains no assignment information.")
            q=(f"Who was assigned the {unrelated}?" if arm.startswith("unrelated")
               else f"Who was assigned the {a}?")
            for rotate in (0,1):
                first,second=((x,y) if rotate==0 else (y,x))
                prefix=(f"Evidence: {src}\nQuestion: {q}\n"
                        f"A) {first}\nB) {second}\nC) NOT STATED\nAnswer:")
                correct=("C" if arm.startswith("unrelated") or arm=="supported_withheld"
                         else ("A" if (arm=="supported_original")== (rotate==0) else "B"))
                out.append({"topic":i,"arm":arm,"rotate":rotate,"prompt":prefix,
                            "correct":correct,"first_label":"A" if rotate==0 else "B",
                            "second_label":"B" if rotate==0 else "A"})
    if len(out)!=80:raise RuntimeError("HOLD incomplete design")
    return out

def preflight(model_name,out):
    if sha_file(ROOT/"EXP051-PROTOCOL-2026-10-16.md")!=PROTOCOL_SHA:
        raise RuntimeError("HOLD protocol edited")
    if out.exists() or out.parent.resolve()!=ROOT.resolve():
        raise RuntimeError("HOLD output exists or outside research")
    if available_gib()<38:raise RuntimeError("HOLD MemAvailable<38GiB")
    p=MODELS[model_name]
    for f in ("model.safetensors.index.json","config.json","tokenizer.json"):
        if not (p/f).is_file():raise RuntimeError("HOLD missing "+str(p/f))
    manifest=json.loads((p/"model.safetensors.index.json").read_text())
    for fname in set(manifest["weight_map"].values()):
        if "/" in fname or not (p/fname).is_file() or (p/fname).stat().st_size<1000:
            raise RuntimeError("HOLD weight shard missing")
    return p,{f:sha_file(p/f) for f in ("config.json","model.safetensors.index.json","tokenizer.json")}
def score(model,tok,rows):
    single={}
    for label in "ABC":
        ids=tok.encode(" "+label,add_special_tokens=False)
        if len(ids)!=1:raise RuntimeError("HOLD label not single token "+label)
        single[label]=ids[0]
    ret=[]
    begin=time.monotonic()
    for j,r in enumerate(rows):
        if j%10==0 and available_gib()<22:
            raise RuntimeError("HOLD MemAvailable<22GiB during eval")
        tokens=tok(r["prompt"],return_tensors="pt",add_special_tokens=True)
        if not 15<=tokens["input_ids"].shape[1]<190:
            raise RuntimeError("HOLD bad input tokens")
        tokens.pop("token_type_ids",None)
        with torch.inference_mode():
            logits=model(**tokens,use_cache=False).logits[0,-1].float()
        ll=torch.log_softmax(logits[[single[c] for c in "ABC"]],dim=0)
        scores={c:float(ll[k]) for k,c in enumerate("ABC")}
        pred=max(scores,key=scores.get)
        m=scores[r["first_label"]]-scores[r["second_label"]]
        ret.append({"topic":r["topic"],"arm":r["arm"],"rotate":r["rotate"],
                    "correct":r["correct"],"prediction":pred,"margin":m})
        del logits,ll,tokens
        if (j+1)%16==0:
            print("PROGRESS",j+1,"of",len(rows),"elapsed_s",round(time.monotonic()-begin,1),flush=True)
    return ret,round(time.monotonic()-begin,3)

def summarize(rows):
    if len(rows)!=80:raise RuntimeError("HOLD incomplete outcome")
    keys={(r["topic"],r["arm"],r["rotate"]) for r in rows}
    if len(keys)!=80:raise RuntimeError("HOLD duplicate outcome")
    strict={}
    for arm in ARMS:
        cell=[r for r in rows if r["arm"]==arm]
        if len(cell)!=16:raise RuntimeError("HOLD missing arm "+arm)
        strict[arm]={"correct":sum(r["correct"]==r["prediction"] for r in cell),
                     "n":len(cell)}
    lookup={(r["topic"],r["arm"],r["rotate"]):r["margin"] for r in rows}
    pairs=[]
    for i in range(8):
        ds=[];di=[]
        for rotate in (0,1):
            s=lookup[i,"supported_original",rotate]-lookup[i,"supported_swapped",rotate]
            u=lookup[i,"unrelated_original",rotate]-lookup[i,"unrelated_swapped",rotate]
            ds.append(s);di.append(s-u)
        pairs.append((statistics.mean(ds),statistics.mean(di)))
    sw=[x[0] for x in pairs];di=[x[1] for x in pairs]
    return {"strict":strict,"source_switch":{"mean":statistics.mean(sw),
            "positive_topics":sum(x>0 for x in sw),"n_topics":8},
            "question_x_evidence":{"mean":statistics.mean(di),
            "positive_topics":sum(x>0 for x in di),"n_topics":8}}
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",choices=tuple(MODELS),required=True)
    ap.add_argument("--execute",action="store_true")
    ap.add_argument("--out",type=Path)
    args=ap.parse_args()
    rows=make()
    fixture_sha=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
    print("EXP051_DESIGN_OK",len(rows),"receipt",fixture_sha,flush=True)
    if not args.execute:
        print("DRY_RUN_NO_MODEL");return
    if args.out is None:raise RuntimeError("HOLD required research report --out")
    p,source_sha=preflight(args.model,args.out)
    from transformers import AutoTokenizer,AutoModelForCausalLM,Qwen3_5ForConditionalGeneration
    tok=AutoTokenizer.from_pretrained(str(p),local_files_only=True,
                                       trust_remote_code=(args.model=="k2"))
    typ=AutoModelForCausalLM if args.model=="k2" else Qwen3_5ForConditionalGeneration
    t=time.monotonic()
    model=typ.from_pretrained(str(p),local_files_only=True,trust_remote_code=(args.model=="k2"),
                              dtype=torch.bfloat16,use_safetensors=True,
                              low_cpu_mem_usage=True)
    model.eval().to("cpu").requires_grad_(False)
    print("EXP051_MODEL_LOADED",args.model,"seconds",round(time.monotonic()-t,1),flush=True)
    try:observations,duration=score(model,tok,rows)
    finally:
        del model,tok;gc.collect()
    result={"protocol_sha256":PROTOCOL_SHA,"model":args.model,"fixture_sha256":fixture_sha,
            "sources_sha256":source_sha,"forwards":len(observations),"seconds":duration,
            "note":"synthetic nonindependent pilot; no human-adjudicated semantic grounding",
            "summary":summarize(observations)}
    args.out.open("x").write(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("EXP051_COMPLETE",json.dumps(result["summary"],sort_keys=True),flush=True)
if __name__=="__main__":main()
