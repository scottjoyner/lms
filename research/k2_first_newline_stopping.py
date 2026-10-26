#!/usr/bin/env python3
"""EXP042: preregistered newline-limited free generation, same EXP041B fixture.

Research-only, no training/deployment, model weights always frozen.
"""
import os
for key,value in {"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1",
                  "CUDA_VISIBLE_DEVICES":"-1","TOKENIZERS_PARALLELISM":"false",
                  "OMP_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"1",
                  "MKL_NUM_THREADS":"2"}.items():os.environ[key]=value
import argparse,gc,hashlib,time
from transformers import StoppingCriteria,StoppingCriteriaList
from pathlib import Path
import torch
from k2_interpolation_pilot import (MID,RL,TOK_SHA,IDX_SHA,CONFIG_SHA,
    preflight,check_shard_layout,sha,available_gib)
from k2_novel_source_fixture import (
    build,prompt,score,score_group,corpus_receipt,RECORDS,CASES,MODES,MAX_NEW)
ROOT=Path("/home/scott/git/wt-knowledge-k2-latex-20261014/50-Research/llm-research/latent-world-geometry-2026")
PROTOCOL_SHA="f5459e7c1c382db2f322d9c4ffd47889d99848c7945e1d6c7ad79d310f6eab4c"
B_PROTOCOL_SHA="634892caa5fdc3870b0001ed4ebf9b8db26c26eeb446a8ccdd2368d0b4b505f4"
FIXTURE_SHA="30366b9ee63bc309a3633da0beb50daa66413f918a99299593d78e6122fe3f81"
PROTOCOL_042="ac9e19750cfe660bf2933715160985cf6f3c70c52fa3cb991617f2d95a930ee9"
BASELINE_041B_SHA="2254874a1aa7e3ac2b63d7f2295bc1aed3b92ba3fba5473c09a5ffd1992f43f4"
torch.set_num_threads(2)

def frozen_sources():
    # A model-free preflight; only execute path enforces 36 GiB admission.
    for i,root in enumerate((MID,RL)):
        if not root.is_dir() or root.is_symlink():
            raise RuntimeError("HOLD missing or redirected checkpoint")
        if sha(root/"tokenizer.json")!=TOK_SHA:
            raise RuntimeError("HOLD tokenizer SHA")
        if sha(root/"model.safetensors.index.json")!=IDX_SHA[i]:
            raise RuntimeError("HOLD weight inventory SHA")
        if sha(root/"config.json")!=CONFIG_SHA[i]:
            raise RuntimeError("HOLD config SHA")
    if sha(ROOT/"EXPERIMENT-041-PROTOCOL.md")!=PROTOCOL_SHA:
        raise RuntimeError("HOLD frozen protocol changed")
    cases=build()
    if corpus_receipt(cases)!=FIXTURE_SHA:
        raise RuntimeError("HOLD fixture content changed")
    return cases

class StopAtGeneratedNewline(StoppingCriteria):
    """Stop on first new-line byte in DECODED new tokens, never prompt text."""
    def __init__(self,tok,prefix):
        self.tokenizer=tok
        self.prefix=prefix
        self.hit=False
    def __call__(self,input_ids,scores,**kwargs):
        if input_ids.ndim!=2 or input_ids.shape[0]!=1 or input_ids.shape[1]<self.prefix:
            raise RuntimeError("HOLD malformed stopping-criteria token history")
        suffix=self.tokenizer.decode(input_ids[0,self.prefix:].tolist(),skip_special_tokens=True)
        self.hit="\n" in suffix
        return bool(self.hit)

def generate_one(model,tok,case):
    if available_gib()<22:
        raise RuntimeError("HOLD live available RAM <22 GiB")
    inputs=tok(prompt(case),return_tensors="pt",truncation=False,
               add_special_tokens=True)
    inputs.pop("token_type_ids",None)
    input_size=int(inputs["input_ids"].shape[1])
    if not 20<input_size<650:
        raise RuntimeError(f"HOLD unexpected prompt token count {input_size}")
    pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    stopper=StopAtGeneratedNewline(tok,input_size)
    with torch.inference_mode():
        out=model.generate(**inputs,do_sample=False,num_beams=1,
           max_new_tokens=MAX_NEW,use_cache=True,
           stopping_criteria=StoppingCriteriaList([stopper]),
           pad_token_id=pad,eos_token_id=tok.eos_token_id,
           return_dict_in_generate=True,output_scores=True)
    sequences=out.sequences
    if sequences.ndim!=2 or sequences.shape[0]!=1:
        raise RuntimeError("HOLD bad generation shape")
    all_ids=sequences[0,input_size:].tolist()
    if not 1<=len(all_ids)<=MAX_NEW or len(out.scores)!=len(all_ids):
        raise RuntimeError("HOLD bad generation length")
    # The scores are complete-vocabulary logits for each genuine free-gen step.
    if any(not torch.isfinite(v).all().item() for v in out.scores):
        raise RuntimeError("HOLD nonfinite generation scores")
    eos=tok.eos_token_id
    if isinstance(eos,(tuple,list)):
        eos=set(eos)
    else:
        eos={eos}
    stop=next((j for j,v in enumerate(all_ids) if v in eos),len(all_ids))
    decoded=tok.decode(all_ids[:stop],skip_special_tokens=True)
    hit_newline="\n" in decoded
    if bool(stopper.hit) and not hit_newline:
        raise RuntimeError("HOLD newline stopper and decoded output disagree")
    if hit_newline and not stopper.hit:
        raise RuntimeError("HOLD newline observed without stopping")
    visible=decoded.split("\n",1)[0] if hit_newline else decoded
    native_eos=stop<len(all_ids)
    cap=(stop>=MAX_NEW and not hit_newline and not native_eos)
    result=score(case,visible,hitcap=cap)
    result["newline_stop"]=int(hit_newline)
    result["native_eos"]=int(native_eos)
    return result,input_size,len(all_ids)

def render(cases,scores,tokens,seconds):
    lines=[
      "# EXP-042 — First-newline stopping on the same authored source fixture",
      "",
      "**2026-10-15 | Measured real pinned Mid2 K2 native CPU generation, research only.**",
      "",
      "## Evidence and source custody",
      "",
      f"- EXP041 original frozen protocol SHA256: `{PROTOCOL_SHA}`.",
      f"- EXP041B baseline preregistration SHA256: `{B_PROTOCOL_SHA}`.",
      f"- EXP042 separately frozen newline-stop protocol SHA256: `{PROTOCOL_042}`.",
      f"- Historical EXP041B full-generate baseline report SHA256: `{BASELINE_041B_SHA}`.",
      "- Original EXP041 attempt stopped before any generation due to invalid >60-token sanity bound.",
      f"- Deterministic authored 8-ledger/24-case fixture SHA256: `{FIXTURE_SHA}`.",
      f"- Mid2 model index SHA256: `{IDX_SHA[0]}`.",
      f"- Mid2 config SHA256: `{CONFIG_SHA[0]}`.",
      f"- Tokenizer SHA256: `{TOK_SHA}`.",
      "- Native K2 base model has no verified official chat template; plain identical prompt for both modes.",
      "- Same exact Mid2 BF16 weights; default RoPE and native RL YaRN rotary module only.",
      "- 48 real greedy free-generation forwards: 24 per RoPE, at most 16 new tokens each.",
      "- Sole decoder intervention: stop at first DECODED generated newline; score text preceding newline.",
      "- Same exact EXP041B source prompts and strict scoring, no semantic postprocessing.",
      "- This is newly generated synthetic source text, **not a human-annotated natural-source heldout benchmark**.",
      "- The authored facts postdate these checkpoint snapshots, but the evaluation makes **no guarantee about training independence, generalization or semantic reliability**.",
      "",
      "## Exact-answer and abstention evaluation",
      "",
      "| RoPE | Case | N | Full strict | First line exact | Gold substring (lenient) | Said exact UNKNOWN | First newline | Native EOS | Hit 16-token cap | Empty |",
      "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for mode in MODES:
        for kind in CASES:
            d=scores[mode][kind]
            lines.append(f"| {mode} | {kind} | {d['n']} | {d['strict']} | "+
             f"{d['first_line']} | {d['contains_gold']} | {d['said_unknown']} | "+
             f"{d['newline_stop']} | {d['native_eos']} | {d['hit_cap']} | {d['empty']} |")
    lines +=[
      "",
      "## Interpretive limits",
      "",
      "- Each source carries a synthetic code, two plausible agents and multiple objects.",
      "- Answerable responses require the full nonce proper name; exact UNKNOWN",
      "  is required for unsupported action, redacted passage and swapped source.",
      "- A substring present in longer completions is reported **leniently**,",
      "  and does not count as an exact factual answer.",
      "- Nonabstention for unsupported questions is format/source-fidelity failure,",
      "  not proof of factual hallucination outside this synthetic fixture.",
      "- The fixture has only eight authored source clusters; two questions",
      "  from the same cluster are not independent observations.",
      "- Model pretraining, prompt-format priors, long-context and real human",
      "  answer quality remain unverified; free generation is not a release gate.",
      "- No model write, merged checkpoint, graph, NAS, routing, credential,",
      "  network topology, promotion or deployment changes occurred.",
      "",
      f"- Prompt tokens min/max across conditions: {min(tokens)} / {max(tokens)}.",
      f"- Total timed generate operation elapsed: {seconds:.2f} seconds.",
      "- Same source and gold labels as EXP041B; any strict change is a response-policy difference.",
      "- No raw text responses, logits, activation tensors, examples or per-item",
      "  measurements were saved to this report.",
      "",
    ]
    return "\n".join(lines)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute",action="store_true")
    ap.add_argument("--out",type=Path)
    args=ap.parse_args()
    cases=frozen_sources()
    if sha(ROOT/"EXPERIMENT-041B-PROTOCOL.md")!=B_PROTOCOL_SHA:
        raise RuntimeError("HOLD EXP041B baseline protocol SHA mismatch")
    if sha(ROOT/"EXPERIMENT-042-PROTOCOL.md")!=PROTOCOL_042:
        raise RuntimeError("HOLD EXP042 frozen protocol SHA mismatch")
    if sha(ROOT/"EXPERIMENT-041B-RESULTS-2026-10-15.md")!=BASELINE_041B_SHA:
        raise RuntimeError("HOLD historical EXP041B result changed")
    print("EXP042_PREFLIGHT_FIXTURE_OK",len(cases),"topics",RECORDS,
          "digest",FIXTURE_SHA[:12],flush=True)
    if not args.execute:
        print("EXP042_DRY_RUN_NO_MODEL_LOADING_OR_WRITES",flush=True)
        return
    if args.out is None or args.out.exists() or args.out.parent.resolve()!=ROOT.resolve():
        raise RuntimeError("HOLD requires novel isolated Markdown report path")
    idx,_,_=preflight() # Exact original 36 GiB RAM gate retained.
    if check_shard_layout(*idx)!=1078285824:
        raise RuntimeError("HOLD 255-tensor shape/parameter inventory")
    from transformers import AutoConfig,AutoTokenizer,AutoModelForCausalLM
    tok=AutoTokenizer.from_pretrained(str(MID),local_files_only=True,trust_remote_code=False)
    cfg=AutoConfig.from_pretrained(str(RL),local_files_only=True,trust_remote_code=True)
    if cfg.rope_parameters["rope_type"]!="yarn":
        raise RuntimeError("HOLD unexpected YaRN config")
    lengths=[int(tok(prompt(row),return_tensors="pt",truncation=False,add_special_tokens=True)["input_ids"].shape[1]) for row in cases]
    if len(lengths)!=24 or not all(20<x<650 for x in lengths):
        raise RuntimeError("HOLD EXP041B full prompt-length preflight")
    print("EXP042_ALL_24_PROMPTS_TOKENIZED",min(lengths),max(lengths),flush=True)
    if available_gib()<36:raise RuntimeError("HOLD preload available RAM")
    model=AutoModelForCausalLM.from_pretrained(str(MID),local_files_only=True,
       trust_remote_code=True,dtype=torch.bfloat16,use_safetensors=True,low_cpu_mem_usage=True)
    model.eval().to("cpu").requires_grad_(False)
    backbone=model.model
    native=backbone.rotary_emb
    ya=type(native)(cfg,device=torch.device("cpu"))
    if native.rope_type!="default" or ya.rope_type!="yarn":
        raise RuntimeError("HOLD mismatched rotary settings")
    collected={};lengths=[];start=time.monotonic()
    try:
        for mode,rot in (("default",native),("yarn",ya)):
            backbone.rotary_emb=rot
            scored=[]
            for j,case in enumerate(cases):
                val,n,start_out=generate_one(model,tok,case)
                scored.append(val);lengths.append(n)
                if (j+1)%4==0:
                    print("EXP042_PROGRESS",mode,j+1,"/",len(cases),
                      "elapsed",round(time.monotonic()-start,1),flush=True)
            if len(scored)!=24:raise RuntimeError("HOLD incomplete source mode")

            grouped=score_group(cases,scored)
            for category in CASES:
                matching=[row for row,val in zip(cases,scored) if row["kind"]==category]
                grouped[category]["newline_stop"]=sum(row["newline_stop"] for row,val in zip(scored,cases) if val["kind"]==category)
                grouped[category]["native_eos"]=sum(row["native_eos"] for row,val in zip(scored,cases) if val["kind"]==category)
                total=grouped[category]["newline_stop"]+grouped[category]["native_eos"]+grouped[category]["hit_cap"]
                if total!=grouped[category]["n"]:
                    raise RuntimeError("HOLD unexpected stopping reason tally")
            collected[mode]=grouped
            print("EXP042_MODE_COMPLETE",mode,[(k,collected[mode][k]["strict"])
                  for k in CASES],flush=True)
        if len(collected)!=2 or len(lengths)!=48:
            raise RuntimeError("HOLD incomplete 48 full generations")
        outcome=render(cases,collected,lengths,time.monotonic()-start)
        args.out.open("x",encoding="utf8").write(outcome)
        print("EXP042_COMPLETE",hashlib.sha256(outcome.encode()).hexdigest(),flush=True)
    finally:
        backbone.rotary_emb=native
        del model,ya
        gc.collect()
if __name__=="__main__":main()
