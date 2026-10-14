#!/usr/bin/env python3
"""EXP023: exact pinned K2 checkpoint weight drift, layer x tensor family.

Read-only CPU tensor scan; markdown-only observations. Not CKA, not teacher proof.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from safetensors import safe_open
import torch

STAGES=("pretrain600k","mid1_75k","mid2_47684","rl_merged","mopd_249")
KINDS=("input_layernorm","post_attention_layernorm","attn.q_proj",
       "attn.k_proj","attn.v_proj","attn.o_proj","mlp.gate_proj",
       "mlp.up_proj","mlp.down_proj")
MODES={k:("self_attn."+k[5:] if k.startswith("attn.") else k)
       for k in KINDS}
REFERENCE_SHA="3818a421a9b9cbe6f52b0a27d8c3da69be0ee1bd4d50071c190d50ba95ee893a"
ROOT=Path("/home/scott/git")
PREFIX=".latent-k2-"
SUFFIX="-research"
torch.set_num_threads(2)

def sha_bytes(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for buf in iter(lambda:f.read(1024*1024),b""):h.update(buf)
    return h.hexdigest()

def open_manifest(root,reference,stage):
    source=root/(PREFIX+stage+SUFFIX)
    if not source.is_dir():raise ValueError("Absent staged K2 model "+stage)
    rows=json.loads((source/"model.safetensors.index.json").read_text())
    keys=rows["weight_map"]
    if len(keys)!=255:raise ValueError("Unexpected model tensor inventory "+stage)
    config=json.loads((source/"config.json").read_text())
    if (config["hidden_size"],config["num_hidden_layers"],config["vocab_size"])!=(1536,28,64256):
        raise ValueError("Unexpected K2 architecture "+stage)
    revisions=reference["checkpoint_revisions"]
    if len(revisions[stage])!=40:raise ValueError("Bad source revision")
    stage_meta=reference.get("intermediate_config_and_index_receipts",{}).get(stage)
    if stage_meta:
        for label,p in (("sha256_config",source/"config.json"),
                        ("sha256_index",source/"model.safetensors.index.json")):
            if sha_bytes(p)!=stage_meta[label]:
                raise ValueError("Source metadata drift "+stage+" "+label)
        if stage_meta["revision"]!=revisions[stage]:
            raise ValueError("Source model commit changed")
    shards={name:source/name for name in set(keys.values())}
    if any(not x.is_file() for x in shards.values()):
        raise ValueError("Missing source tensor shard "+stage)
    return {"root":source,"tensor_to_file":keys,"shards":shards,
            "revision":revisions[stage],
            "index_sha256":sha_bytes(source/"model.safetensors.index.json"),
            "config_sha256":sha_bytes(source/"config.json"),
            "shard_sha256":{name:sha_bytes(path)
                            for name,path in sorted(shards.items())}}

def tname(layer,kind):
    if kind in ("input_layernorm","post_attention_layernorm"):
        return f"model.layers.{layer}.{kind}.weight"
    prefix=MODES[kind]
    return f"model.layers.{layer}.{prefix}.weight"

def tensor_pair(man1,man2,name):
    path1=man1["shards"][man1["tensor_to_file"][name]]
    path2=man2["shards"][man2["tensor_to_file"][name]]
    with safe_open(str(path1),framework="pt",device="cpu") as first, \
         safe_open(str(path2),framework="pt",device="cpu") as second:
        x=first.get_tensor(name)
        y=second.get_tensor(name)
        if x.dtype!=torch.bfloat16 or y.dtype!=torch.bfloat16 or x.shape!=y.shape:
            raise ValueError("Tensor dtype/shape mismatch "+name)
        xx=x.float().reshape(-1)
        yy=y.float().reshape(-1)
        dx=yy-xx
        if not bool(torch.isfinite(dx).all()):
            raise ValueError("Non-finite model parameter "+name)
        nx=float(torch.dot(xx,xx).item())
        ny=float(torch.dot(yy,yy).item())
        nd=float(torch.dot(dx,dx).item())
        dot=float(torch.dot(xx,yy).item())
        changed=int(torch.count_nonzero(x!=y).item())
        equal=changed==0
        n=int(xx.numel())
    if nx<=0 or ny<=0:raise ValueError("Degenerate tensor norm "+name)
    return dict(nx=nx,ny=ny,nd=nd,dot=dot,n=n,changed=changed,equal=equal)

def merge(rows):
    nx=sum(x["nx"] for x in rows);ny=sum(x["ny"] for x in rows)
    nd=sum(x["nd"] for x in rows);dot=sum(x["dot"] for x in rows)
    count=sum(x["n"] for x in rows)
    return dict(n=count,cosine=dot/math.sqrt(nx*ny),
                delta_l2=math.sqrt(nd/nx),
                same_tensors=sum(x["equal"] for x in rows),tensors=len(rows),
                changed_fraction=sum(x["changed"] for x in rows)/count)

def render(entries,info,reference_digest,duration):
    lines=["# EXP-023 — K2 checkpoint weight drift across all 28 layers",
      "",
      "**October 14, 2026 | new direct BF16 checkpoint measurement | LMS frozen read-only scan.**",
      "This is not representational CKA or a model training/teacher proof.",
      "",
      "## Exact source receipts","",
      f"EXP018 reference experiment SHA256: `{reference_digest}`.",
      "All checkpoint revisions are pinned and all 255-tensor indices checked.",
      "The model weights were read from existing isolated CPU files; no weights,",
      "per-parameter values, activations or prompts were saved to knowledge.",
      "",
      "| Stage | HF commit | Index SHA256 |",
      "|---|---|---|"]
    for stage in STAGES:
        meta=info[stage]
        lines.append(f"| {stage} | `{meta['revision']}` | `{meta['index_sha256']}` |")
    lines +=["","## Full local checkpoint shard custody","",
      "These per-file SHA256 digests identify the exact locally measured",
      "Safetensors bytes. The published revision tags and index hashes",
      "supply the upstream source identity; no downloaded file changed.",
      "",
      "| Stage | Safetensors shard | SHA256 |",
      "|---|---|---|"]
    for stage in STAGES:
        for file,digest in sorted(info[stage]["shard_sha256"].items()):
            lines.append(f"| {stage} | `{file}` | `{digest}` |")
    lines +=["","## Transition-level drift across 252 transformer-layer tensors","",
       "Each layer has nine parameter families (Q/K/V/O, MLP gate/up/down,",
       "and two layernorms), totaling 252 distinct tensors.",
       "Whole-transformer cosine is computed across original weight coordinates;",
       "relative L2 = ||W_next−W_previous||₂ / ||W_previous||₂.",
       "",
       "| Transition | Parameter-weighted cosine | Relative L2 drift | Changed parameter fraction | Exactly identical tensors / 252 |",
       "|---|---:|---:|---:|---:|"]
    for edge,x in entries.items():
        val=merge([v for row in x["by_layer"].values() for v in row.values()])
        lines.append(f"| {edge} | {val['cosine']:.8f} | {val['delta_l2']:.6f} | "
          f"{val['changed_fraction']*100:.2f}% | {val['same_tensors']}/252 |")
    lines+=["","## Per-stage per-layer full-weight relative drift","",
      "Layer comparisons average **parameters**, not tensor labels. High weight",
      "cosine does not imply the same function; rescaling, RoPE and nonlinear",
      "network interactions are not controlled by this metric.",
      ""]
    for edge,x in entries.items():
        lines +=[f"### {edge}","",
          "| Transformer layer | Param-weighted cosine | Relative L2 drift | Exact tensors / 9 |",
          "|---:|---:|---:|---:|"]
        for layer in range(28):
            s=merge(list(x["by_layer"][str(layer)].values()))
            lines.append(f"| {layer} | {s['cosine']:.8f} | {s['delta_l2']:.6f} | {s['same_tensors']}/9 |")
        lines+=["","#### Tensor-family decomposition","",
          "| Parameter family | Weighted cosine | Relative L2 drift |",
          "|---|---:|---:|"]
        for kind in KINDS:
            m=merge([x["by_layer"][str(layer)][kind] for layer in range(28)])
            lines.append(f"| {kind} | {m['cosine']:.8f} | {m['delta_l2']:.6f} |")
        lines+=[""]
    lines+=["## Scientific interpretation and limits","",
      "This experiment measures parameter-space continuity **in the same model",
      "architecture**, unlike cross-model CKA. It neither proves semantic",
      "behavior preserved nor isolates causal effects of RL or distillation.",
      "The EXP-018 CKA trajectory and EXP-019/020/021 behavioral readouts are",
      "separate observations and should be compared as separate quantities.",
      "Full BF16 main↔mOPD 255/255 tensor identity was independently proven",
      "in EXP-018 and does not require reloading a sixth checkpoint here.",
      "",
      "Sequential frozen tensor access, two CPU threads; elapsed time is logged",
      "to stdout but intentionally omitted from reproducible report content.",
      "Model files were not modified; source tensors are never written to output.",
      "Human ratings, generalized competence and teacher identity remain unverified.",
      "",
      "## Reproduce","",
      "```bash",
      "PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 /home/scott/git/.latent-qwen35-venv/bin/python research/k2_weight_drift.py --evidence-dir /home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026 --out /home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026/EXPERIMENT-023-WEIGHT-DRIFT.md",
      "```","",
      "Requires local official pinned checkpoint files and scottjoyner/lms",
      "research branch `research/k2-layer-matrix-20261014`. Existing output",
      "is never overwritten unless `--replace` is explicitly requested.",""]
    return "\n".join(lines)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence-dir",type=Path,required=True)
    p.add_argument("--model-root",type=Path,default=ROOT)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--replace",action="store_true")
    a=p.parse_args()
    if a.out.exists() and not a.replace:raise SystemExit("STOP: output already exists")
    if not a.out.parent.exists():raise SystemExit("STOP: output directory missing")
    ref=a.evidence_dir/"results_exp018_intermediate_k2_20261014.json"
    if sha_bytes(ref)!=REFERENCE_SHA:raise ValueError("Source EXP018 digest drift")
    reference=json.loads(ref.read_text())
    info={s:open_manifest(a.model_root,reference,s) for s in STAGES}
    if any(set(info[s]["tensor_to_file"])!=set(info[STAGES[0]]["tensor_to_file"])
           for s in STAGES[1:]):
        raise ValueError("Different K2 parameter naming across stages")
    begin=time.monotonic()
    entries={}
    for before,after in zip(STAGES,STAGES[1:]):
        edge=f"{before} → {after}"
        layers={}
        for layer in range(28):
            if os.path.exists("/proc/meminfo"):
                mem_kb=next(int(line.split()[1]) for line in
                    Path("/proc/meminfo").read_text().splitlines()
                    if line.startswith("MemAvailable:"))
                if mem_kb/1024/1024<20:raise RuntimeError("Host RAM gate <20GiB")
            stats={kind:tensor_pair(info[before],info[after],tname(layer,kind))
                   for kind in KINDS}
            layers[str(layer)]=stats
            if layer in (0,13,27):
                print("WEIGHT_LAYER",edge,layer,round(merge(list(stats.values()))["delta_l2"],6),flush=True)
        entries[edge]={"by_layer":layers}
        print("WEIGHT_STAGE",edge,merge([v for row in layers.values() for v in row.values()]),flush=True)
    duration=time.monotonic()-begin
    out=render(entries,info,REFERENCE_SHA,duration)
    tmp=a.out.with_name(a.out.name+".tmp")
    if tmp.exists():raise RuntimeError("Output temp collision")
    tmp.write_text(out,encoding="utf8")
    tmp.replace(a.out)
    print("EXP023_PASS",a.out,len(out.splitlines()),hashlib.sha256(out.encode()).hexdigest(),flush=True)

if __name__=="__main__":main()
