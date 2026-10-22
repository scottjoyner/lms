#!/usr/bin/env python3
"""EXP035: offline ephemeral same-topology K2 interpolation (no weight artifacts).

Pinned Mid2/RL checkpoint sources stay read-only. All interpolation happens
in memory. A small AUTHORED, NON-INDEPENDENT QA fixture is exploratory only.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
for key, value in {"HF_HUB_OFFLINE":"1", "TRANSFORMERS_OFFLINE":"1",
                   "CUDA_VISIBLE_DEVICES":"-1", "TOKENIZERS_PARALLELISM":"false",
                   "OMP_NUM_THREADS":"2", "OPENBLAS_NUM_THREADS":"1",
                   "MKL_NUM_THREADS":"2"}.items():
    os.environ[key] = value
import torch
from safetensors import safe_open

MID = Path("/home/scott/git/.latent-k2-mid2_47684-research")
RL = Path("/home/scott/git/.latent-k2-rl_merged-research")
TOK_SHA = "7a42bd219df0b747f000ca3e39f8eca9b93e965707e8be0b6603cac56aaad738"
IDX_SHA = ("3cfa718ec080cc86f41b6f6406a54598fe681a6b4bd9176f278b9d01bf77284d",
           "7c8e777b3a44329d3228a33e11e2008fc64691748ece8fb775e6bd8e097e57b5")
CONFIG_SHA = ("f754a7b359c7428713f2ffd3da4e503ff3b09cd40dc93c28cdc7d05412396974",
              "3a4585b64269fbd4851c585cce0606a80cf1a0d1b88af031e7a7d770c6e6b87d")
ALPHAS=(0.0,0.25,0.5,0.75,1.0)
EXPECTED_PARAMS=1078285824
# Eight hand-authored small input records, NOT independent study or SQuAD holdout.
# Answer C denotes source-withheld/UNKNOWN; symmetric option order is necessary.
STIMULI = (
 ("Mira placed a copper key in the east cabinet. Leon placed a brass key in the west cabinet.",
  "Who placed the copper key?", "Mira", "Leon", "A"),
 ("Mira placed a copper key in the east cabinet. Leon placed a brass key in the west cabinet.",
  "Who placed the brass key?", "Mira", "Leon", "B"),
 ("Nora painted the red sign. Felix painted the blue sign.",
  "Who painted the blue sign?", "Nora", "Felix", "B"),
 ("Nora painted the red sign. Felix painted the blue sign.",
  "Who painted the red sign?", "Nora", "Felix", "A"),
 ("The report mentions a silver box on a high shelf. No person is named.",
  "Who moved the silver box?", "Nora", "Felix", "C"),
 ("The memo describes an open gate but does not mention who unlocked it.",
  "Who unlocked the gate?", "Mira", "Leon", "C"),
 ("Iris stacked the green folder. Theo stacked the yellow folder.",
  "Who stacked the yellow folder?", "Iris", "Theo", "B"),
 ("Iris stacked the green folder. Theo stacked the yellow folder.",
  "Who stacked the green folder?", "Iris", "Theo", "A"),
)
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def available_gib():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])/1024**2
    raise RuntimeError("No meminfo")
def preflight():
    if available_gib()<36:
        raise RuntimeError("HOLD: less than 36 GiB available")
    indices=[]
    for j,root in enumerate((MID,RL)):
        if not root.is_dir() or root.is_symlink():
            raise RuntimeError("HOLD: missing or redirected checkpoint root")
        if sha(root/"tokenizer.json")!=TOK_SHA:
            raise RuntimeError("HOLD: changed tokenizer")
        if sha(root/"model.safetensors.index.json")!=IDX_SHA[j]:
            raise RuntimeError("HOLD: changed weight index")
        if sha(root/"config.json")!=CONFIG_SHA[j]:
            raise RuntimeError("HOLD: changed config")
        idx=json.loads((root/"model.safetensors.index.json").read_text())
        if len(idx["weight_map"])!=255 or len(set(idx["weight_map"].values()))!=28:
            raise RuntimeError("HOLD: stage inventory mismatch")
        if any("/" in v or not v.endswith(".safetensors") or not (root/v).is_file()
               for v in idx["weight_map"].values()):
            raise RuntimeError("HOLD: missing, unsafe or redirected shard")
        indices.append(idx["weight_map"])
    if set(indices[0])!=set(indices[1]):
        raise RuntimeError("HOLD: name mismatch")
    a,b=[json.loads((r/"config.json").read_text()) for r in (MID,RL)]
    for key in ("model_type","num_hidden_layers","hidden_size","intermediate_size",
                "vocab_size","num_attention_heads","num_key_value_heads","head_dim",
                "num_experts","tie_word_embeddings"):
        if a.get(key)!=b.get(key):
            raise RuntimeError("HOLD: incompatible "+key)
    if (a.get("num_experts"),a.get("num_hidden_layers"),a.get("hidden_size"))!=(0,28,1536):
        raise RuntimeError("HOLD: unexpected architecture")
    return indices, a, b
def check_shard_layout(index_a,index_b):
    """Exhaustive shape and dtype test without materializing all weights."""
    total=0
    for name in sorted(index_a):
        with safe_open(str(MID/index_a[name]),framework="pt",device="cpu") as a:
            x=a.get_slice(name)
            shape_a=x.get_shape()
            dtype_a=x.get_dtype()
        with safe_open(str(RL/index_b[name]),framework="pt",device="cpu") as b:
            y=b.get_slice(name)
            shape_b=y.get_shape()
            dtype_b=y.get_dtype()
        if shape_a!=shape_b or dtype_a!=dtype_b or dtype_a!="BF16":
            raise RuntimeError("HOLD: incompatible shape/dtype "+name)
        count=1
        for d in shape_a:count*=d
        total+=count
    if total!=EXPECTED_PARAMS:
        raise RuntimeError("HOLD: wrong scalar count "+str(total))
    return total

def interpolate_tensor(base, target, alpha):
    """Round only once at BF16 output; alpha=0/1 are byte-identical endpoints."""
    if alpha==0: return base
    if alpha==1: return target
    if base.shape!=target.shape or base.dtype!=torch.bfloat16 or target.dtype!=base.dtype:
        raise ValueError("interpolation shape/dtype")
    if not torch.isfinite(base).all() or not torch.isfinite(target).all():
        raise ValueError("HOLD: nonfinite source weights")
    return torch.lerp(base.to(torch.float32),target.to(torch.float32),alpha).to(torch.bfloat16)
def stage_weights(model,index,alpha):
    """Interpolate into *ephemeral* CPU model; never save, register, or deploy."""
    named=dict(model.named_parameters())
    if set(named)!=set(index):
        raise RuntimeError("HOLD: model names differ from pinned checkpoint")
    if alpha==0:
        return 255
    with torch.no_grad():
        for name in sorted(index):
            if available_gib()<22:
                raise RuntimeError("HOLD: <22 GiB during staging")
            p=named[name]
            with safe_open(str(RL/index[name]),framework="pt",device="cpu") as src:
                t=src.get_tensor(name)
            if tuple(p.shape)!=tuple(t.shape) or p.dtype!=torch.bfloat16:
                raise RuntimeError("HOLD: model load mismatch "+name)
            if not torch.isfinite(t).all():
                raise RuntimeError("HOLD: nonfinite target "+name)
            if alpha==1:
                p.copy_(t)
            else:
                p.copy_(interpolate_tensor(p.detach(),t,alpha))
            del t
    return len(named)

def make_prompts():
    return [
      ("Source: "+source+"\nQuestion: "+question+
       "\nA) "+a+"\nB) "+b+
       "\nC) NOT STATED\nChoose A, B, or C from source only.\nAnswer:", gold)
      for source,question,a,b,gold in STIMULI
    ]
def evaluate(model,tokenizer):
    pairs=make_prompts()
    candidate_ids=[]
    for option in (" A"," B"," C"):
        ids=tokenizer.encode(option,add_special_tokens=False)
        if len(ids)!=1: raise RuntimeError("HOLD: candidate is not 1 token")
        candidate_ids.extend(ids)
    correct=0; nll=0.; margin=0.; c_correct=0
    for prompt,gold in pairs:
        if available_gib()<22:raise RuntimeError("HOLD: memory during eval")
        toks=tokenizer(prompt,return_tensors="pt",add_special_tokens=True,
                       truncation=False)
        toks.pop("token_type_ids",None)
        if toks["input_ids"].shape[1]>190:
            raise RuntimeError("HOLD: excessive prompt tokens")
        with torch.inference_mode():
            logits=model(**toks,use_cache=False).logits[0,-1,candidate_ids].float()
            prob=torch.softmax(logits,dim=-1)
        g={"A":0,"B":1,"C":2}[gold]
        pred=int(torch.argmax(prob))
        correct+=int(pred==g)
        if gold=="C":c_correct+=int(pred==g)
        nll+=-float(torch.log(prob[g].clamp_min(1e-30)))
        margin+=float(prob[g]-torch.max(torch.cat((prob[:g],prob[g+1:]))))
        del logits,prob,toks
    n=len(pairs)
    return {"n":n,"strict":correct,"unknown_strict":c_correct,
            "mean_choice_nll":round(nll/n,6),
            "mean_choice_margin":round(margin/n,6)}

def run_stage(alpha, idx):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import gc
    t0=time.monotonic()
    if available_gib()<36:raise RuntimeError("HOLD: insufficient pre-stage RAM")
    # Always read Mid2, so every alpha is evaluated with the identical default RoPE.
    tok=AutoTokenizer.from_pretrained(str(MID),local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(
       str(MID),local_files_only=True,trust_remote_code=True,
       dtype=torch.bfloat16,use_safetensors=True,low_cpu_mem_usage=True)
    model.eval().to("cpu").requires_grad_(False)
    n=stage_weights(model,idx,alpha)
    scores=evaluate(model,tok)
    scores.update(alpha=alpha,tensors=n,elapsed_s=round(time.monotonic()-t0,2))
    del model,tok
    gc.collect()
    return scores
def manuscript_report(info, rows):
    ia,ib,aconf,bconf=info
    lines=[
      "# EXP-035 — Ephemeral K2 Mid2/RL BF16 interpolation pilot",
      "",
      "**Research-only 2026-10-15 | Authored synthetic microfixture, not independent validation.**",
      "No checkpoints were written, registered, trained, or deployed.",
      "",
      "## Verified checkpoint custody",
      "",
      f"- Mid2 index SHA256: `{IDX_SHA[0]}`; config SHA256: `{CONFIG_SHA[0]}`.",
      f"- RL index SHA256: `{IDX_SHA[1]}`; config SHA256: `{CONFIG_SHA[1]}`.",
      f"- Shared tokenizer SHA256: `{TOK_SHA}` (identical).",
      f"- {EXPECTED_PARAMS:,} parameters, 255 named tensors, 28 shards each; BF16 shape/dtype checked.",
      "- All candidates run with **Mid2 default RoPE**, fixed tokenizer and identical authored prompts.",
      "- Native RL YaRN differs, so alpha=1 here is NOT its native published-config score.",
      "- Mixing performed in float32 and cast once to BF16 in memory; endpoints unmodified.",
      "",
      "## Measured exploratory 8-prompt results",
      "",
      "| Alpha | Correct /8 | UNKNOWN correct /2 | Choice NLL ↓ | Choice margin ↑ | Seconds |",
      "|---:|---:|---:|---:|---:|---:|"]
    for x in rows:
        lines.append(f"| {x['alpha']:.2f} | {x['strict']}/8 | {x['unknown_strict']}/2 | "+
                     f"{x['mean_choice_nll']:.6f} | {x['mean_choice_margin']:.6f} | {x['elapsed_s']:.1f} |")
    lines +=["",
      "**Do not treat differences as downstream improvements.** The prompts are eight authored "+
      "short questions (three A, three B, two C); output compares single-token A/B/C "+
      "conditional logits, not free generation, calibrated confidence, reasoning "+
      "or independently held-out semantics. No CI, causal-neuron discovery, "+
      "teacher lineage, or production gate is established.",
      "",
      "## Method",
      "Code: LMS research/k2_interpolation_pilot.py on isolated research branch. "+
      "Preflight is non-mutating; execute requires explicit flag; writing to a "+
      "new report path fails closed. No output tensors are persisted.",
      "Models remain NO-GO for production grounded QA. No production effects.",
      ""]
    return "\n".join(lines)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute",action="store_true",
                   help="actually load frozen weights and run small local forwards")
    p.add_argument("--out",type=Path,
                   help="new isolated research Markdown output; must not exist")
    args=p.parse_args()
    torch.set_num_threads(2)
    ind,ac,bc=preflight()
    count=check_shard_layout(*ind)
    print("PREFLIGHT_OK",count,"tensors",len(ind[0]),
          "shared tokenizer",TOK_SHA[:12],"fixed_rope","mid2/default",flush=True)
    if not args.execute:
        print("DRY_RUN_ONLY: nothing loaded, merged or written",flush=True)
        return
    if args.out is None or args.out.exists():
        raise RuntimeError("HOLD: --out must be a new file")
    if args.out.parent.resolve()!=Path(
        "/home/scott/git/wt-knowledge-k2-latex-20261014/50-Research/llm-research/latent-world-geometry-2026").resolve():
        raise RuntimeError("HOLD: report must target isolated research knowledge worktree")
    result=[]
    for alpha in ALPHAS:
        print("START_ALPHA",alpha,flush=True)
        data=run_stage(alpha,ind[1]);result.append(data)
        print("FINISHED_ALPHA",alpha,
              "strict",data["strict"],"nll",data["mean_choice_nll"],flush=True)
    text=manuscript_report((ind[0],ind[1],ac,bc),result)
    args.out.open("x",encoding="utf-8").write(text)
    print("WROTE_RESEARCH_REPORT",args.out,flush=True)
if __name__=="__main__":
    main()