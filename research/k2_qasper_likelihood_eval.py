#!/usr/bin/env python3
"""EXP048 real teacher-forced K2 log-likelihood evidence discrimination."""
import os
for k,v in {"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1",
            "CUDA_VISIBLE_DEVICES":"-1","TOKENIZERS_PARALLELISM":"false",
            "OMP_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"1",
            "MKL_NUM_THREADS":"2"}.items():os.environ[k]=v
import argparse,gc,hashlib,math,statistics,time
from pathlib import Path
import torch
from k2_interpolation_pilot import (
 MID,RL,TOK_SHA,IDX_SHA,CONFIG_SHA,preflight,check_shard_layout,sha,available_gib)
from k2_qasper_likelihood_fixture import (
 inputs,digest,token_inventory,CATEGORIES,ARMS,MODES)
ROOT=Path("/home/scott/git/wt-knowledge-k2-latex-20261014/50-Research/llm-research/latent-world-geometry-2026")
PROTOCOL_SHA="fda02f3046519613f2e83804c3d88dda0f8f2e69761b3d2e7cfcb85c94f761ce"
COHORT_SHA="1c4f7f148f33c1e38868f40a58a70f2858276cb41571b15d872e015268e395e5"
PRIOR_SHA="636e87bb8312077995e1a8f85e0c812387147fecc290142cab458860b5e92640"
torch.set_num_threads(2)

def frozen_preflight():
    if sha(ROOT/"EXPERIMENT-048-PROTOCOL-2026-10-16.md")!=PROTOCOL_SHA:
        raise RuntimeError("HOLD EXP048 registered protocol SHA changed")
    if sha(ROOT/"EXPERIMENT-047-RESULTS-2026-10-15.md")!=PRIOR_SHA:
        raise RuntimeError("HOLD EXP047 archived report changed")
    for i,dir in enumerate((MID,RL)):
        if not dir.is_dir() or dir.is_symlink():
            raise RuntimeError("HOLD missing/symlink model root")
        for name,exp in (("model.safetensors.index.json",IDX_SHA[i]),
                         ("config.json",CONFIG_SHA[i]),
                         ("tokenizer.json",TOK_SHA)):
            if sha(dir/name)!=exp:
                raise RuntimeError("HOLD model checkpoint/rotation SHA changed")
    cases=inputs()
    if len(cases)!=32 or digest(cases)!=COHORT_SHA:
        raise RuntimeError("HOLD source/candidate selection drift")
    return cases

def candidate_logprob(model,prefix,answer):
    if available_gib()<22:
        raise RuntimeError("HOLD RAM below original 22GiB in-step floor")
    full=prefix+answer
    if not 1<=len(answer)<=32 or not 20<len(prefix)<750:
        raise RuntimeError("HOLD candidate or input length")
    ids=torch.tensor([full],dtype=torch.long)
    with torch.inference_mode():
        logits=model(input_ids=ids,use_cache=False,return_dict=True).logits
        if logits.ndim!=3 or logits.shape[0]!=1 or logits.shape[1]!=len(full):
            raise RuntimeError("HOLD model output shape")
        window=logits[0,len(prefix)-1:len(prefix)-1+len(answer),:].float()
        if window.shape[0]!=len(answer) or not torch.isfinite(window).all().item():
            raise RuntimeError("HOLD nonfinite teacher-forced logits")
        selected=torch.log_softmax(window,dim=-1).gather(
           1,torch.tensor(answer,dtype=torch.long).unsqueeze(-1)).squeeze(-1)
        vals=selected.tolist()
    if any(not math.isfinite(v) or v>0.0001 for v in vals):
        raise RuntimeError("HOLD invalid token conditional likelihood")
    return {"mean":sum(vals)/len(vals),"sum":sum(vals),"n":len(vals)}

def aggregate(cases,mode_outputs):
    if len(mode_outputs)!=2 or any(len(mode_outputs[m])!=32 for m in MODES):
        raise RuntimeError("HOLD incomplete model forwards")
    groups=[];pairs=[]
    for mode in MODES:
        vals=mode_outputs[mode]
        for kind in CATEGORIES:
            for arm in ARMS:
                a=[v for item,v in zip(cases,vals)
                   if item["kind"]==kind and item["arm"]==arm]
                if len(a)!=8:
                    raise RuntimeError("HOLD wrong eight-paper group")
                means=[v["margin_mean"] for v in a]
                sums=[v["margin_sum"] for v in a]
                groups.append({"mode":mode,"kind":kind,"arm":arm,"n":8,
                 "avg_margin_mean":statistics.mean(means),
                 "median_margin_mean":statistics.median(means),
                 "avg_margin_sum":statistics.mean(sums),
                 "answer_favored":sum(x>0 for x in means),
                 "unknown_favored":sum(x<0 for x in means),
                 "ties":sum(x==0 for x in means)})
            change=[]
            for topic in range(8):
                matched=[v["margin_mean"] for item,v in zip(cases,vals)
                        if item["kind"]==kind and item["topic"]==topic]
                if len(matched)!=2:raise RuntimeError("HOLD broken paired study")
                # Input order is original then withheld for every category/topic.
                change.append(matched[0]-matched[1])
            pairs.append({"mode":mode,"kind":kind,"n":8,
                          "avg_source_minus_withheld":statistics.mean(change),
                          "median_source_minus_withheld":statistics.median(change),
                          "positive":sum(x>0 for x in change),
                          "negative":sum(x<0 for x in change),
                          "ties":sum(x==0 for x in change)})
    if len(groups)!=8 or len(pairs)!=4:raise RuntimeError("HOLD group counts")
    return groups,pairs

def report(groups,pairs,prompt_lengths,choice_lengths,elapsed):
    lines=["# EXP-048 — Teacher-forced candidate evidence discrimination",
      "", "**2026-10-16 | 128 native frozen K2 CPU forward passes; research only.**",
      "", "## Preregistered source and model custody", "",
      "- Frozen pre-model protocol SHA256: "+PROTOCOL_SHA,
      "- Official reused EXP047 8-paper QASPER question/gold receipt: 0f9c4fad7ba905a3b1283911ea50b77f4470b710183509425e24bd14328f69c5.",
      "- Paired source/withheld question-and-candidate receipt: "+COHORT_SHA,
      "- Untouched EXP047 strict free-generation report SHA256: "+PRIOR_SHA,
      "- The same 8 public human-labeled full-paper QASPER source clusters were REUSED.",
      "- Only 16 questions: 8 extractive-supported and 8 human-FULL-PAPER-unanswerable.",
      "- Original evidence versus a fixed withheld passage; identical questions, paper titles",
      "  and two prechosen response strings across source contexts.",
      "- Candidate A is the human extractive gold phrase from the supported question,",
      "  prefixed with one ASCII space. Candidate B is literal ' UNKNOWN'.",
      "- The gold-phrase candidate used for an unanswerable question is an unrelated",
      "  same-paper foil, NOT a human-adjudicated non-entailing answer.",
      "- Same Mid2 BF16 1,078,285,824-parameter checkpoint, identical frozen learned",
      "  weights under native default or RL-config YaRN rotary modules.",
      "- Exactly 64 contexts x 2 teacher-forced candidates = 128 actual model forwards.",
      "- NO free generation, no choice samples, no calibrated answer probability.",
      "", "## Candidate conditional-likelihood contrasts", "",
      "- M = mean(target-token logP(human-reference-or-foil)) - mean(target-token logP(' UNKNOWN')).",
      "- Positive M favors the supplied short phrase; negative M favors UNKNOWN.",
      "- Total-token log-likelihood contrast is included separately because candidate",
      "  token counts differ and average token NLL can mask length bias.",
      "", "| RoPE | Human category | Evidence | N | Mean M (nats/token) | Median M | Mean total-logP diff | Answer favored | UNKNOWN favored | Ties |",
      "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for a in groups:
        lines.append(f"| {a['mode']} | {a['kind']} | {a['arm']} | {a['n']} | "
          f"{a['avg_margin_mean']:.6f} | {a['median_margin_mean']:.6f} | "
          f"{a['avg_margin_sum']:.6f} | {a['answer_favored']} | "
          f"{a['unknown_favored']} | {a['ties']} |")
    lines+=["", "## Matched per-question evidence effect", "",
      "- Contrast: M(original evidence) - M(withheld source), same question and candidates.",
      "", "| RoPE | Human category | N | Mean evidence effect | Median evidence effect | Positive | Negative | Ties |",
      "|---|---|---:|---:|---:|---:|---:|---:|"]
    for d in pairs:
        lines.append(f"| {d['mode']} | {d['kind']} | {d['n']} | "
           f"{d['avg_source_minus_withheld']:.6f} | {d['median_source_minus_withheld']:.6f} | "
           f"{d['positive']} | {d['negative']} | {d['ties']} |")
    lines+=["", "## Methodological limitations and safety", "",
      "- This task gives K2 the HUMAN GOLD ANSWER as candidate text before evaluating",
      "  model likelihood: positive scores do not prove free-form answer ability.",
      "- Different target token lengths and contextual continuation priors strongly confound",
      "  margins. Mean per-token and total log-likelihood are both provided.",
      "- Source effects in model likelihood are descriptive and do not establish",
      "  semantic understanding, calibration or neuron-level mechanism.",
      "- Eight public papers repeatedly evaluated; possible pretraining exposure,",
      "  no independent blinded replication or statistical significance claim.",
      "- Human unanswerable labels refer to full QASPER papers; a single paragraph",
      "  may not be sufficient context to fairly infer answerability.",
      f"- Prompt tokens min/max: {min(prompt_lengths)}/{max(prompt_lengths)}.",
      f"- Candidate target tokens min/max: {min(choice_lengths)}/{max(choice_lengths)}.",
      f"- Timed actual model passes elapsed: {elapsed:.2f} seconds.",
      "- No copyrighted excerpt, QID, paper title, raw logits, per-case decision,",
      "  raw candidate text, model checkpoint or tensor saved in this report.",
      "- Both PRs draft/open/unmerged. NO serving/merging/promotion, training,",
      "  credentials/network/routing, production graph/NAS, deployment or arXiv.",
      ""]
    return "\n".join(lines)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--execute",action="store_true")
    ap.add_argument("--out",type=Path)
    args=ap.parse_args()
    rows=frozen_preflight()
    print("EXP048_FROZEN_SOURCE_OK",len(rows),COHORT_SHA[:12],flush=True)
    if not args.execute:
        print("EXP048_DRY_RUN_NO_MODEL",flush=True)
        return
    if args.out is None or args.out.exists() or args.out.parent.resolve()!=ROOT.resolve():
        raise RuntimeError("HOLD output exists or outside isolated research")
    idx,_,_=preflight()
    if check_shard_layout(*idx)!=1078285824:
        raise RuntimeError("HOLD 255-tensor learned weight inventory")
    from transformers import AutoTokenizer,AutoConfig,AutoModelForCausalLM
    tok=AutoTokenizer.from_pretrained(str(MID),trust_remote_code=False,local_files_only=True)
    token_sets=token_inventory(tok,rows)
    prompt_lengths=[len(x[0]) for x in token_sets]
    candidate_lengths=[len(v) for tup in token_sets for v in tup[1:]]
    print("EXP048_TOKENIZER_READY",min(prompt_lengths),max(prompt_lengths),
          min(candidate_lengths),max(candidate_lengths),flush=True)
    if available_gib()<36:raise RuntimeError("HOLD RAM before K2 load <36GiB")
    cfg=AutoConfig.from_pretrained(str(RL),trust_remote_code=True,local_files_only=True)
    if cfg.rope_parameters["rope_type"]!="yarn":raise RuntimeError("HOLD YaRN topology")
    model=AutoModelForCausalLM.from_pretrained(str(MID),trust_remote_code=True,
      local_files_only=True,dtype=torch.bfloat16,use_safetensors=True,
      low_cpu_mem_usage=True)
    model.eval().to("cpu").requires_grad_(False)
    original=model.model.rotary_emb
    yarn=type(original)(cfg,device=torch.device("cpu"))
    if original.rope_type!="default" or yarn.rope_type!="yarn":
        raise RuntimeError("HOLD unexpected rotary modules")
    measures={};beg=time.monotonic();steps=0
    try:
        for mode,rot in (("default",original),("yarn",yarn)):
            model.model.rotary_emb=rot
            scores=[]
            for i,(prefix,cand_answer,cand_unknown) in enumerate(token_sets):
                a=candidate_logprob(model,prefix,cand_answer);steps+=1
                b=candidate_logprob(model,prefix,cand_unknown);steps+=1
                scores.append({"margin_mean":a["mean"]-b["mean"],
                               "margin_sum":a["sum"]-b["sum"]})
                if (i+1)%4==0:
                    print("EXP048_PROGRESS",mode,i+1,"/",len(rows),
                          "passes",steps,"elapsed",round(time.monotonic()-beg,1),
                          flush=True)
            measures[mode]=scores
        if steps!=128 or digest(rows)!=COHORT_SHA:
            raise RuntimeError("HOLD incomplete forwards or mutable source")
        groups,pairs=aggregate(rows,measures)
        result=report(groups,pairs,prompt_lengths,candidate_lengths,
                      time.monotonic()-beg)
        args.out.open("x",encoding="utf8").write(result)
        print("EXP048_COMPLETE_SHA",hashlib.sha256(result.encode()).hexdigest(),flush=True)
    finally:
        model.model.rotary_emb=original
        del model,yarn
        gc.collect()
if __name__=="__main__":main()
