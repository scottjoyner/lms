#!/usr/bin/env python3
"""EXP049 equal-length, both-in-evidence QASPER gold/foil forced likelihood."""
import os
for k,v in {"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","CUDA_VISIBLE_DEVICES":"-1",
 "TOKENIZERS_PARALLELISM":"false","OMP_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"1",
 "MKL_NUM_THREADS":"2"}.items():os.environ[k]=v
import argparse,gc,hashlib,math,statistics,time
from pathlib import Path
import torch
from k2_interpolation_pilot import MID,RL,sha,preflight,check_shard_layout,available_gib
from k2_qasper_likelihood_eval import candidate_logprob,frozen_preflight as old_frozen
from k2_qasper_matched_foil_fixture import make,source_digest,MODES,ARMS
from k2_qasper_decode_fixture import prompt

ROOT=Path("/home/scott/git/wt-knowledge-k2-latex-20261014/50-Research/llm-research/latent-world-geometry-2026")
PROTOCOL_SHA="3456fe74178410faeed6d4bcab9eef80cc547c1d80e04740f7ad322fc4e10ce6"
COHORT_SHA="16cd90495c18a1cb27492aca0a2ffb8b2def35eb6c615e7c952ed230af149b4d"
EXP048_SHA="7b34a7eac6918cd587a4fcc3a976fefd2377e48de84414f72cfd268d6f6d8bd0"
torch.set_num_threads(2)

def frozen():
    if sha(ROOT/"EXPERIMENT-049-PROTOCOL-2026-10-16.md")!=PROTOCOL_SHA:
        raise RuntimeError("HOLD frozen EXP049 protocol drift")
    if sha(ROOT/"EXPERIMENT-048-RESULTS-2026-10-16.md")!=EXP048_SHA:
        raise RuntimeError("HOLD EXP048 canonical report drift")
    old_frozen() # validates EXP048 source/model/tokenizer and archived EXP047
    cases=make()
    if len(cases)!=16 or source_digest(cases)!=COHORT_SHA:
        raise RuntimeError("HOLD matched candidate receipt")
    return cases

def grouped(cases,res):
    if any(len(res.get(m,[]))!=16 for m in MODES):
        raise RuntimeError("HOLD partial native study")
    cells=[];deltas=[]
    for mode in MODES:
        for arm in ARMS:
            vals=[v for c,v in zip(cases,res[mode]) if c["arm"]==arm]
            if len(vals)!=8:raise RuntimeError("HOLD count per source arm")
            margins=[x["sum_margin"] for x in vals]
            norm=[x["mean_margin"] for x in vals]
            cells.append({"mode":mode,"arm":arm,"n":8,
               "mean_sum":statistics.mean(margins),
               "median_sum":statistics.median(margins),
               "mean_per_token":statistics.mean(norm),
               "gold_favored":sum(v>0 for v in margins),
               "foil_favored":sum(v<0 for v in margins),
               "ties":sum(v==0 for v in margins)})
        pairs=[]
        for i in range(0,16,2):
            if cases[i]["arm"]!="original" or cases[i+1]["arm"]!="withheld":
                raise RuntimeError("HOLD input pairing order")
            pairs.append(res[mode][i]["sum_margin"]-res[mode][i+1]["sum_margin"])
        deltas.append({"mode":mode,"n":8,"mean_source_delta":statistics.mean(pairs),
          "median_source_delta":statistics.median(pairs),
          "positive":sum(v>0 for v in pairs),
          "negative":sum(v<0 for v in pairs),
          "ties":sum(v==0 for v in pairs)})
    return cells,deltas

def report(cells,deltas,lengths,counts,elapsed):
    lines=["# EXP-049 — Matched target-token-length, same-evidence foil control",
      "", "**2026-10-16 | Actual 64 frozen K2 BF16 CPU teacher-forced passes; draft research.**",
      "", "## Registered custody", "",
      "- Pre-inference frozen protocol SHA256: "+PROTOCOL_SHA,
      "- Same exact eight reused public QASPER supported paper topics as EXP047/048.",
      "- Whole 16-context prompt/gold/foil receipt SHA256: "+COHORT_SHA,
      "- Canonical preserved EXP048 report SHA256: "+EXP048_SHA,
      "- Gold and foil source strings are different contiguous phrases BOTH present",
      "  in the same human-supported passage, with identical candidate token counts",
      "  per question, one leading ASCII space, disjoint [a-z]{3,} content words.",
      "- Fixed deterministic foil selection (char length, word count, SHA),",
      "  no model-score-guided candidate engineering or subsequent tuning.",
      "- Same frozen Mid2 BF16 learned weights under default and RL-derived YaRN.",
      "- 8 questions x source original/withheld x 2 RoPE x 2 strings = 64 forwards.",
      "", "## Exact token-count-matched total log-likelihood contrast", "",
      "- M = sum target token logP(human gold) - sum target token logP(foil).",
      "- Gold and foil have exactly the same token counts, so normalizing",
      "  both by their common length cannot change which candidate is favored.",
      "", "| RoPE | Passage | N | Mean total M (nats) | Median total M | Mean M per token | Gold favored | Foil favored | Ties |",
      "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for x in cells:
        lines.append(f"| {x['mode']} | {x['arm']} | {x['n']} | {x['mean_sum']:.6f} | "
            f"{x['median_sum']:.6f} | {x['mean_per_token']:.6f} | "
            f"{x['gold_favored']} | {x['foil_favored']} | {x['ties']} |")
    lines+=["", "## Per-paper source-minus-withheld preference shift", "",
      "| RoPE | N | Mean source effect | Median source effect | Positive | Negative | Ties |",
      "|---|---:|---:|---:|---:|---:|---:|"]
    for x in deltas:
        lines.append(f"| {x['mode']} | {x['n']} | {x['mean_source_delta']:.6f} | "
          f"{x['median_source_delta']:.6f} | {x['positive']} | {x['negative']} | {x['ties']} |")
    lines+=["", "## Interpretation and safety boundaries", "",
      "- Gold text was GIVEN to the likelihood evaluator, NOT generated by the model.",
      "- Foils are source-present lexical alternatives, not independent human verified",
      "  semantically unanswerable distractors; syntax/frequency can still favor one.",
      "- A positive source difference is compatible with both source-semantic",
      "  recognition and generic passage/continuation priming.",
      "- No independent heldout QA, model calibration, K2/Qwen teacher identity",
      "  or production readiness can be inferred from eight reused public papers.",
      f"- Prompt input token range: {min(lengths)}..{max(lengths)}.",
      f"- Common gold/foil candidate token range: {min(counts)}..{max(counts)}.",
      f"- Timed native inference: {elapsed:.2f} seconds.",
      "- No raw paper passage, foil string, prompt, token logits, per-paper result",
      "  or learned weights archived. No training/merge/serving/routing, graph/NAS,",
      "  credentials/network, deployment or public manuscript submission.",""]
    return "\n".join(lines)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--execute",action="store_true")
    ap.add_argument("--out",type=Path);args=ap.parse_args()
    rows=frozen();print("EXP049_FROZEN_OK",len(rows),COHORT_SHA[:12],flush=True)
    if not args.execute:
        print("EXP049_DRY_RUN_NO_MODEL",flush=True);return
    if args.out is None or args.out.exists() or args.out.parent.resolve()!=ROOT.resolve():
        raise RuntimeError("HOLD isolated unique outcome file")
    idx,_,_=preflight()
    if check_shard_layout(*idx)!=1078285824:raise RuntimeError("HOLD model parameter topology")
    from transformers import AutoConfig,AutoTokenizer,AutoModelForCausalLM
    tok=AutoTokenizer.from_pretrained(str(MID),trust_remote_code=False,local_files_only=True)
    bundle=[]
    for x in rows:
        p=tok.encode(prompt(x["row"]),add_special_tokens=True)
        a=tok.encode(x["candidate_gold"],add_special_tokens=False)
        b=tok.encode(x["candidate_foil"],add_special_tokens=False)
        if not 20<len(p)<750 or len(a)!=len(b) or len(a)!=x["n"] or not 1<=len(a)<=32:
            raise RuntimeError("HOLD token-equivalent foil preflight")
        bundle.append((p,a,b))
    print("EXP049_TOKENIZER_OK",min(len(x[0]) for x in bundle),
          max(len(x[0]) for x in bundle),flush=True)
    if available_gib()<36:raise RuntimeError("HOLD premodel available RAM <36GiB")
    cfg=AutoConfig.from_pretrained(str(RL),trust_remote_code=True,local_files_only=True)
    if cfg.rope_parameters["rope_type"]!="yarn":raise RuntimeError("HOLD YaRN rope config")
    model=AutoModelForCausalLM.from_pretrained(str(MID),trust_remote_code=True,
        local_files_only=True,dtype=torch.bfloat16,use_safetensors=True,low_cpu_mem_usage=True)
    model.eval().to("cpu").requires_grad_(False)
    original=model.model.rotary_emb
    yarn=type(original)(cfg,device=torch.device("cpu"))
    if original.rope_type!="default" or yarn.rope_type!="yarn":
        raise RuntimeError("HOLD model rope module")
    out={};steps=0;start=time.monotonic()
    try:
        for mode,rope in (("default",original),("yarn",yarn)):
            model.model.rotary_emb=rope
            scored=[]
            for i,(pre,gold,foil) in enumerate(bundle):
                a=candidate_logprob(model,pre,gold);steps+=1
                b=candidate_logprob(model,pre,foil);steps+=1
                if a["n"]!=b["n"]:raise RuntimeError("HOLD matched-length drift")
                scored.append({"sum_margin":a["sum"]-b["sum"],
                               "mean_margin":a["mean"]-b["mean"]})
                if (i+1)%4==0:
                    print("EXP049_PROGRESS",mode,i+1,"/",len(bundle),
                          "passes",steps,flush=True)
            out[mode]=scored
        if steps!=64 or source_digest(rows)!=COHORT_SHA:
            raise RuntimeError("HOLD incomplete study or altered source receipt")
        cells,deltas=grouped(rows,out)
        text=report(cells,deltas,[len(x[0]) for x in bundle],
                    [len(x[1]) for x in bundle],time.monotonic()-start)
        args.out.open("x",encoding="utf8").write(text)
        print("EXP049_COMPLETE_SHA",hashlib.sha256(text.encode()).hexdigest(),flush=True)
    finally:
        model.model.rotary_emb=original
        del model,yarn
        gc.collect()
if __name__=="__main__":main()
