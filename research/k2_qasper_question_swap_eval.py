#!/usr/bin/env python3
"""EXP050 source x question difference-in-differences, exact-length candidates."""
import os
for k,v in {"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","CUDA_VISIBLE_DEVICES":"-1",
"OMP_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"2",
"TOKENIZERS_PARALLELISM":"false"}.items():os.environ[k]=v
import argparse,gc,hashlib,statistics,time
from pathlib import Path
import torch
from k2_qasper_question_swap_fixture import build,digest,MODES,KINDS,ARMS
from k2_qasper_matched_foil_eval import frozen as frozen_049
from k2_qasper_likelihood_eval import candidate_logprob
from k2_qasper_decode_fixture import prompt
from k2_interpolation_pilot import MID,RL,preflight,check_shard_layout,sha,available_gib

ROOT=Path("/home/scott/git/wt-knowledge-k2-latex-20261014/50-Research/llm-research/latent-world-geometry-2026")
PROTOCOL_SHA="fe56dc8786eda348ce11c43dfefaaba5bc8db9bdc288c3ed2232d0aebe61dcd0"
COHORT_SHA="fd1101840c2a0b3857b60d236be4f6f9a3f501c7b8d8f68728580b32146957df"
BASE_REPORT_SHA="d500e9f6a72081629c7dd55b057a9ff073c88442e6dfb51ceccd8aec04518463"
torch.set_num_threads(2)

def frozen():
    if sha(ROOT/"EXPERIMENT-050-PROTOCOL-2026-10-16.md")!=PROTOCOL_SHA:
        raise RuntimeError("HOLD preregistered EXP050 protocol")
    if sha(ROOT/"EXPERIMENT-049-RESULTS-2026-10-16.md")!=BASE_REPORT_SHA:
        raise RuntimeError("HOLD source report EXP049 changed")
    frozen_049()
    rows=build()
    if digest(rows)!=COHORT_SHA or len(rows)!=32:
        raise RuntimeError("HOLD cohort prompt and candidate selection drift")
    return rows

def stats(rows,observations):
    if any(len(observations.get(m,[]))!=32 for m in MODES):
        raise RuntimeError("HOLD incomplete 128-forward study")
    groups=[];question=[];difference=[]
    for mode in MODES:
        a=observations[mode]
        lookup={(r["topic"],r["arm"],r["kind"]):v for r,v in zip(rows,a)}
        if len(lookup)!=32:raise RuntimeError("HOLD paper-cell pairing")
        for arm in ARMS:
            for kind in KINDS:
                values=[lookup[(t,arm,kind)] for t in range(8)]
                margins=[v["sum_margin"] for v in values]
                groups.append({"mode":mode,"arm":arm,"kind":kind,
                  "n":8,"mean":statistics.mean(margins),
                  "median":statistics.median(margins),
                  "mean_per_token":statistics.mean([v["mean_margin"] for v in values]),
                  "gold":sum(v>0 for v in margins),
                  "foil":sum(v<0 for v in margins),"ties":sum(v==0 for v in margins)})
            deltas=[lookup[(t,arm,"supported")]["sum_margin"]-
                    lookup[(t,arm,"human_unanswerable")]["sum_margin"]
                    for t in range(8)]
            question.append({"mode":mode,"arm":arm,"n":8,
              "mean":statistics.mean(deltas),"median":statistics.median(deltas),
              "positive":sum(v>0 for v in deltas),
              "negative":sum(v<0 for v in deltas),
              "ties":sum(v==0 for v in deltas)})
        did=[]
        for t in range(8):
            actual=(lookup[(t,"original","supported")]["sum_margin"]-
                    lookup[(t,"original","human_unanswerable")]["sum_margin"])
            absent=(lookup[(t,"withheld","supported")]["sum_margin"]-
                    lookup[(t,"withheld","human_unanswerable")]["sum_margin"])
            did.append(actual-absent)
        difference.append({"mode":mode,"n":8,
           "mean":statistics.mean(did),"median":statistics.median(did),
           "positive":sum(v>0 for v in did),
           "negative":sum(v<0 for v in did),
           "ties":sum(v==0 for v in did)})
    return groups,question,difference

def base_repeat_gate(groups):
    # Frozen published original matched-foil data (EXP049) from source report.
    expected={("default","original"):(8.129787,6.854238,8),
       ("default","withheld"):(9.428242,7.285981,6),
       ("yarn","original"):(8.790408,8.480912,8),
       ("yarn","withheld"):(11.698724,7.836914,8)}
    for (mode,arm),(mean,median,gold) in expected.items():
        cell=next(x for x in groups if x["mode"]==mode and
              x["arm"]==arm and x["kind"]=="supported")
        if abs(cell["mean"]-mean)>0.000003 or abs(cell["median"]-median)>0.000003 or cell["gold"]!=gold:
            raise RuntimeError("HOLD baseline EXP049 source/question replay mismatch")

def report(groups,question,difference,lens,counts,seconds):
    lines=["# EXP-050 — Question relevance x source presence, matched-length foil",
      "", "**2026-10-16 | All 128 real K2 Mid2 BF16 candidate forwards; research-only.**",
      "", "## Frozen design provenance", "",
      "- Frozen protocol SHA256: "+PROTOCOL_SHA,
      "- Exact 32-prompt question-source-candidate digest: "+COHORT_SHA,
      "- Full EXP049 immutable previous report SHA256: "+BASE_REPORT_SHA,
      "- Same eight public QASPER papers and both in-source, equal-token-length",
      "  gold/alternative phrase candidates from EXP049; no new hidden examples.",
      "- Human supported question versus DIFFERENT original human FULL-PAPER",
      "  unanswerable question from SAME paper, with candidates unchanged.",
      "- Original source paragraph versus fixed withheld-source placeholder.",
      "- Frozen Mid2 learned BF16 weights under default versus native YaRN rotary.",
      "- 8 papers x 2 question kinds x 2 sources x 2 rotations x 2 strings",
      "  = 128 real native CPU teacher-forced candidate forward passes.",
      "- All four baseline EXP049 supported-question source cells exactly",
      "  reproduced original mean, median and gold/foil preference counts.",
      "", "## Gold vs matched-length foil whole sequence conditional likelihood", "",
      "- M = sum target-token logP(gold) - sum target-token logP(foil).",
      "- Identical gold and foil token counts per source article.",
      "", "| RoPE | Evidence | Question | N | Mean M | Median M | Mean per-token M | Gold favored | Foil favored | Ties |",
      "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for d in groups:
        lines.append(f"| {d['mode']} | {d['arm']} | {d['kind']} | {d['n']} | "
            f"{d['mean']:.6f} | {d['median']:.6f} | {d['mean_per_token']:.6f} | "
            f"{d['gold']} | {d['foil']} | {d['ties']} |")
    lines+=["", "## Within-source question-change effects", "",
      "- Qeffect(source)=M(supported question, source)-M(unanswerable question,source).",
      "", "| RoPE | Source | N | Mean Qeffect | Median Qeffect | Positive | Negative | Ties |",
      "|---|---|---:|---:|---:|---:|---:|---:|"]
    for d in question:
        lines.append(f"| {d['mode']} | {d['arm']} | {d['n']} | {d['mean']:.6f} | "
           f"{d['median']:.6f} | {d['positive']} | {d['negative']} | {d['ties']} |")
    lines+=["", "## Prespecified question x evidence difference-in-differences", "",
      "- D = Qeffect(original source) - Qeffect(withheld source), by paper.",
      "", "| RoPE | Papers | Mean D | Median D | Positive D | Negative D | Ties |",
      "|---|---:|---:|---:|---:|---:|---:|"]
    for d in difference:
        lines.append(f"| {d['mode']} | {d['n']} | {d['mean']:.6f} | {d['median']:.6f} | "
           f"{d['positive']} | {d['negative']} | {d['ties']} |")
    lines+=["","## Boundaries", "",
      "- Semantic entailment of the same-paper alternative phrase was not independently",
      "  adjudicated; wrong-question change is not human-labeled candidate rejection.",
      "- Original full-paper human unanswerable questions receive one short excerpt.",
      "- Public eight-paper sample reused, possible model pretraining leakage,",
      "  no significance or causal neuron attribution, no free generation.",
      "- Human extractive gold answers were PROVIDED to candidate scorer.",
      f"- Tokenized prompt min/max: {min(lens)}..{max(lens)}.",
      f"- Gold/foil matched target-token range: {min(counts)}..{max(counts)}.",
      f"- Timed actual inference: {seconds:.2f} seconds.",
      "- No raw source text, candidate, per-case scores, logits, model parameters",
      "  or checkpoints persisted in GitHub research evidence.",
      "- Both PRs draft/unmerged. No model training, serving, promotion,",
      "  production Neo4j/NAS, credentials/network/routing, app deployment or arXiv.",
      ""]
    return "\n".join(lines)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    rows=frozen()
    print("EXP050_FROZEN_OK",len(rows),COHORT_SHA[:12],flush=True)
    if not args.execute:
        print("EXP050_DRY_RUN_NO_MODEL",flush=True);return
    if args.out is None or args.out.exists() or args.out.parent.resolve()!=ROOT.resolve():
        raise RuntimeError("HOLD unique output path within knowledge only")
    idx,_,_=preflight()
    if check_shard_layout(*idx)!=1078285824:raise RuntimeError("HOLD model tensor count")
    from transformers import AutoTokenizer,AutoConfig,AutoModelForCausalLM
    tok=AutoTokenizer.from_pretrained(str(MID),local_files_only=True,trust_remote_code=False)
    bundle=[]
    for row in rows:
        p=tok.encode(prompt(row["row"]),add_special_tokens=True)
        g=tok.encode(row["gold"],add_special_tokens=False)
        f=tok.encode(row["foil"],add_special_tokens=False)
        if not 20<len(p)<750 or len(g)!=len(f) or len(g)!=row["n"] or not 1<=len(g)<=32:
            raise RuntimeError("HOLD token target parity")
        bundle.append((p,g,f))
    print("EXP050_TOKENIZER_OK",min(len(x[0]) for x in bundle),
          max(len(x[0]) for x in bundle),flush=True)
    if available_gib()<36:raise RuntimeError("HOLD RAM premodel below36")
    cfg=AutoConfig.from_pretrained(str(RL),trust_remote_code=True,local_files_only=True)
    if cfg.rope_parameters["rope_type"]!="yarn":raise RuntimeError("HOLD rotary config")
    model=AutoModelForCausalLM.from_pretrained(str(MID),trust_remote_code=True,
       local_files_only=True,dtype=torch.bfloat16,use_safetensors=True,low_cpu_mem_usage=True)
    model.eval().to("cpu").requires_grad_(False)
    original=model.model.rotary_emb
    yarn=type(original)(cfg,device=torch.device("cpu"))
    if original.rope_type!="default" or yarn.rope_type!="yarn":
        raise RuntimeError("HOLD rotary module topology")
    observations={};count=0;begin=time.monotonic()
    try:
        for mode,rot in (("default",original),("yarn",yarn)):
            model.model.rotary_emb=rot
            scored=[]
            for j,(p,g,f) in enumerate(bundle):
                v=candidate_logprob(model,p,g);count+=1
                w=candidate_logprob(model,p,f);count+=1
                scored.append({"sum_margin":v["sum"]-w["sum"],
                               "mean_margin":v["mean"]-w["mean"]})
                if (j+1)%8==0:
                    print("EXP050_PROGRESS",mode,j+1,"/",len(bundle),
                          "passes",count,"elapsed",round(time.monotonic()-begin,1),flush=True)
            observations[mode]=scored
        if count!=128 or digest(rows)!=COHORT_SHA:
            raise RuntimeError("HOLD incomplete forwards or source drift")
        groups,question,difference=stats(rows,observations)
        base_repeat_gate(groups)
        result=report(groups,question,difference,[len(x[0]) for x in bundle],
                      [len(x[1]) for x in bundle],time.monotonic()-begin)
        args.out.open("x",encoding="utf8").write(result)
        print("EXP050_COMPLETE_SHA",hashlib.sha256(result.encode()).hexdigest(),flush=True)
    finally:
        model.model.rotary_emb=original
        del model,yarn
        gc.collect()
if __name__=="__main__":main()
