#!/usr/bin/env python3
"""EXP024: pinned frozen K2 weight × RoPE 2x2 factorial; Markdown evidence only."""
import argparse
import gc
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

for k,v in {"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1",
            "CUDA_VISIBLE_DEVICES":"-1","OPENBLAS_NUM_THREADS":"1",
            "OMP_NUM_THREADS":"2","MKL_NUM_THREADS":"2"}.items():os.environ[k]=v
import numpy as np
import torch
from transformers import AutoConfig,AutoTokenizer

K2_STAGES=("mid2_47684","rl_merged")
ROPES=("default","yarn")
SOURCE_SHA="3818a421a9b9cbe6f52b0a27d8c3da69be0ee1bd4d50071c190d50ba95ee893a"
WEIGHT_REPORT_SHA="56c4b353462ced77cec1637aa675cb2edd4e1c1190b7f7f2bbe47cec142a69a4"
CORPUS_SHA="a1c4172d25e8430f419a65e150590f376d7b519f45c31b04b8306d6b91309907"
SAMPLED=(0,7,14,21,28)
torch.set_num_threads(2)

def sha(path):
    d=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):d.update(b)
    return d.hexdigest()

def weight_receipts(backbone):
    # Weight identity control across the in-memory positional module swap.
    return {
      "first_attention_output":hashlib.sha256(
          backbone.layers[0].self_attn.o_proj.weight.detach().cpu()
          .contiguous().view(torch.uint16).numpy().tobytes()).hexdigest(),
      "last_mlp_down":hashlib.sha256(
          backbone.layers[27].mlp.down_proj.weight.detach().cpu()
          .contiguous().view(torch.uint16).numpy().tobytes()).hexdigest(),
    }

def exact_source(evidence):
    if sha(evidence/"results_exp018_intermediate_k2_20261014.json")!=SOURCE_SHA:
        raise RuntimeError("EXP018 source SHA changed")
    if sha(evidence/"EXPERIMENT-023-WEIGHT-DRIFT.md")!=WEIGHT_REPORT_SHA:
        raise RuntimeError("EXP023 weight custody changed")
    original=json.loads((evidence/"results_exp018_intermediate_k2_20261014.json").read_text())
    if tuple(original["checkpoint_order"])!=(
        "pretrain600k","mid1_75k","mid2_47684","rl_merged","mopd_249","main"):
        raise RuntimeError("Stage order changed")
    return original

def abstract_source(evidence):
    sys.path.insert(0,str(evidence))
    import exp015_fixture as fixture
    import run_exp015_role_qa as scorer
    import run_exp016_k2_pretrain_release as geometry
    import run_exp018_k2_intermediate_gates as stage
    import run_exp011_qwen35_pilot as cka
    rows=fixture.make()
    if fixture.validate(rows)["source_sha256"]!=CORPUS_SHA:
        raise RuntimeError("Fixture digest drift")
    return fixture,scorer,geometry,stage,cka,rows

def format_cell(v):
    return "—" if v is None else f"{v:.4f}"

def cka_matrix(cka,a,b,pool,contrast):
    arr=[]
    for first in SAMPLED:
        row=[]
        for second in SAMPLED:
            aa=a[first,pool]
            bb=b[second,pool]
            if contrast:
                aa=aa[::2]-aa[1::2]
                bb=bb[::2]-bb[1::2]
            val=cka.centered_cka(aa,bb)
            if val is not None and (not np.isfinite(val) or val<-1e-6 or val>1.00001):
                raise RuntimeError("Invalid CKA cell")
            row.append(val)
        arr.append(row)
    return arr

def markdown_grid(m):
    lines=["| Left \\ Right | "+" | ".join(str(n) for n in SAMPLED)+" |",
           "|---|"+ "|".join(["---:"]*5)+"|"]
    for i,row in enumerate(m):
        lines.append(f"| {SAMPLED[i]} | "+" | ".join(format_cell(v) for v in row)+" |")
    return "\n".join(lines)

def qa_projection(stage,rotary,metrics):
    data=metrics[stage,rotary]
    return data["overall"]["correct"]

def render(stats,activations,pins,cka):
    a=qa_projection
    m2def=a("mid2_47684","default",stats)
    m2ya=a("mid2_47684","yarn",stats)
    rldef=a("rl_merged","default",stats)
    rlya=a("rl_merged","yarn",stats)
    lines=["# EXP-024 — Fixed K2 weights × positional encoding ablation",
           "",
           "**October 14, 2026 | newly measured frozen 2×2 model-forward factorial.**",
           "All models isolated on x1-370 CPU with weights frozen.",
           "The fixture is authored synthetic, not independent human ground truth.",
           "",
           "## Exact source custody","",
           f"EXP018 model/behavior evidence SHA256 `{SOURCE_SHA}`.",
           f"EXP023 complete model shard custody report SHA256 `{WEIGHT_REPORT_SHA}`.",
           f"EXP015 256 QA fixture SHA256 `{CORPUS_SHA}`.",
           "Two official K2 revisions:","",
           "| Weight stage | Pinned checkpoint |",
           "|---|---|"]
    for stage in K2_STAGES:
        lines.append(f"| {stage} | `{pins[stage]}` |")
    lines += ["","## Behavior factorial: 256 paired authored A/B prompts",
        "",
        "| Frozen weights | RoPE implementation | Correct/256 | Actor/128 | Recipient/128 | Answer-A preference |",
        "|---|---|---:|---:|---:|---:|"]
    for stage in K2_STAGES:
        for rope in ROPES:
            rec=stats[stage,rope];overall=rec["overall"]
            lines.append(f"| {stage} | {rope} | {overall['correct']} | "
                         f"{rec['role_type']['actor']['correct']} | "
                         f"{rec['role_type']['target']['correct']} | "
                         f"{100*overall['fraction_prefer_A']:.1f}% |")
    lines += ["","### Controlled differences",
        "",
        f"- Changing only RoPE on **mid2 weights**: {m2ya-m2def:+d}/256 correct.",
        f"- Changing only RoPE on **RL weights**: {rlya-rldef:+d}/256 correct.",
        f"- Changing weights with **default RoPE fixed**: {rldef-m2def:+d}/256 correct.",
        f"- Changing weights with **YaRN fixed**: {rlya-m2ya:+d}/256 correct.",
        f"- Interaction (difference of RoPE effects): "
        f"{(rlya-rldef)-(m2ya-m2def):+d}/256.",
        "",
        "These interventions isolate the *forward-computation* effect of RoPE",
        "at fixed weights on the same prompts. Weight swap compares two trained",
        "checkpoints and cannot isolate actual RL training dynamics.",
        "",
        "## Hidden-state layer-overlap under controlled interventions",
        "",
        "All matrices use the original 64 balanced reversed synthetic facts.",
        "Do not conflate whole-sentence CKA with signed role correctness.",
        "A CKA value is representation geometry, not weight ancestry or",
        "proof of a specific teacher. D→−D sign inversion is invisible to CKA.",
        ""]
    pairs=[
        ("mid2 fixed weights: default→YaRN",("mid2_47684","default"),("mid2_47684","yarn")),
        ("RL fixed weights: default→YaRN",("rl_merged","default"),("rl_merged","yarn")),
        ("default RoPE fixed: mid2→RL",("mid2_47684","default"),("rl_merged","default")),
        ("YaRN fixed: mid2→RL",("mid2_47684","yarn"),("rl_merged","yarn")),
        ("observed native mid2/default→RL/YaRN",("mid2_47684","default"),("rl_merged","yarn")),
    ]
    lines += ["| Controlled comparison | Mean pooled final CKA | Last-token final CKA |",
              "|---|---:|---:|"]
    matrices=[]
    for label,left,right in pairs:
        matmean=cka_matrix(cka,activations[left],activations[right],"mean",False)
        matlast=cka_matrix(cka,activations[left],activations[right],"last",False)
        lines.append(f"| {label} | {format_cell(matmean[-1][-1])} | {format_cell(matlast[-1][-1])} |")
        matrices.append((label,matmean,matlast,left,right))
    for label,matmean,matlast,left,right in matrices:
        lines +=["",f"### {label}","",
                "#### Mean-pooling whole-fact 5×5 CKA","",
                markdown_grid(matmean),"",
                "#### Last-token whole-fact 5×5 CKA","",
                markdown_grid(matlast),""]
        for pool in ("mean","last"):
            role=cka_matrix(cka,activations[left],activations[right],pool,True)
            lines +=[f"#### {pool} signed-role 5×5 CKA","",markdown_grid(role),""]
    lines +=["## Limits and reproduction","",
        "The only in-memory intervention was replacement of the rotary",
        "embedding module; every native answer score and representative",
        "learned-tensor hash was checked against its original state.",
        "No prompts/labels were generated independently by human reviewers.",
        "These are descriptive synthetic counts, not population causal claims.",
        "No model file changes, training, routing, NAS/graph or production effects.",
        "",
        "Reproduce with pinned scottjoyner/lms research branch and the",
        "existing official BF16 checkpoint shards documented by EXP023:",
        "",
        "```bash",
        "python3 research/k2_rope_factorial.py --evidence-dir /home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026 --out /home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026/EXPERIMENT-024-ROPE-FACTORIAL.md",
        "```",""]
    return "\n".join(lines)

def main():
    pa=argparse.ArgumentParser(description=__doc__)
    pa.add_argument("--evidence-dir",type=Path,required=True)
    pa.add_argument("--out",type=Path,required=True)
    pa.add_argument("--replace",action="store_true")
    args=pa.parse_args()
    if args.out.exists() and not args.replace:raise SystemExit("STOP output exists")
    if not args.out.parent.is_dir():raise SystemExit("STOP output parent absent")
    start=time.monotonic()
    import runpy
    sys.path.insert(0,str(args.evidence_dir))
    from run_exp013_k2_qwen_pilot import mem_gib
    if mem_gib()<38:raise RuntimeError("EXP024 memory admission <38GiB")
    prev=exact_source(args.evidence_dir)
    fixture,scorer,geometry,stages,cka,rows=abstract_source(args.evidence_dir)
    facts=geometry.unique_facts(rows)
    base=AutoTokenizer.from_pretrained(str(geometry.BASE),local_files_only=True,
                                        trust_remote_code=False)
    config={r:AutoConfig.from_pretrained(str(stages.stage_path(
            "mid2_47684" if r=="default" else "rl_merged")),
            local_files_only=True,trust_remote_code=True)
            for r in ROPES}
    if (config["default"].rope_parameters["rope_type"]!="default" or
        config["yarn"].rope_parameters["rope_type"]!="yarn"):
        raise RuntimeError("Wrong controlled positional configuration")
    summary={}
    acts={}
    pins={}
    for stage in K2_STAGES:
        model,tok,parity=stages.load_stage(stage,base,rows)
        if parity!=256:raise RuntimeError("Tokenizer stage drift")
        backbone=model.model
        oldrot=backbone.rotary_emb
        native="default" if stage=="mid2_47684" else "yarn"
        before=weight_receipts(backbone)
        head=model.lm_head.weight[scorer.token_ids(tok),:].float().clone()
        pins[stage]=next(x[1] for x in stages.STAGES if x[0]==stage)
        try:
            for rope in ROPES:
                if mem_gib()<22:raise RuntimeError("EXP024 mid-run memory <22GiB")
                alternative=type(oldrot)(config=config[rope],device=torch.device("cpu"))
                if not bool(torch.isfinite(alternative.inv_freq).all()):
                    raise RuntimeError("Non-finite rotary frequencies")
                backbone.rotary_emb=alternative
                record=scorer.measure("k2",backbone,tok,head,rows)
                result=scorer.summary(record)
                if rope==native:
                    expected=prev["behavior_by_stage"][stage]["overall"]
                    if (result["overall"]["correct"]!=expected["correct"] or
                        abs(result["overall"]["fraction_prefer_A"]-
                            expected["fraction_prefer_A"])>0.00001):
                        raise RuntimeError("Native exact baseline behavior failed "+stage)
                summary[stage,rope]=result
                acts[stage,rope]=geometry.extract_k2(backbone,tok,facts)
                print("EXP024_STAGE",stage,rope,
                      result["overall"]["correct"],result["role_type"],
                      flush=True)
                del alternative,record,result
                gc.collect()
        finally:
            backbone.rotary_emb=oldrot
        if weight_receipts(backbone)!=before:
            raise RuntimeError("K2 weights changed during RoPE ablation")
        del model,tok,head,oldrot,backbone
        gc.collect()
    output=render(summary,acts,pins,cka)
    outtmp=args.out.with_name(args.out.name+".tmp")
    if outtmp.exists():raise RuntimeError("Output temporary collision")
    outtmp.write_text(output,encoding="utf8")
    outtmp.replace(args.out)
    print("EXP024_PASS",len(output.splitlines()),
          hashlib.sha256(output.encode()).hexdigest(),
          "duration_seconds",round(time.monotonic()-start,2),
          "peak_rss_gib",round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,3),
          flush=True)

if __name__=="__main__":main()
