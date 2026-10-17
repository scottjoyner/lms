#!/usr/bin/env python3
"""EXP052 MiniCPM-V 4.7-1B text-only BF16 counterfactual smoke comparison."""
import os
for k,v in {"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","CUDA_VISIBLE_DEVICES":"-1",
            "TOKENIZERS_PARALLELISM":"false","OMP_NUM_THREADS":"4","MKL_NUM_THREADS":"4"}.items():
    os.environ[k]=v
import argparse,gc,hashlib,json,time
from pathlib import Path
import torch
from k2_qwen4b_exp051 import make,score,summarize,available_gib,sha_file
ROOT=Path(__file__).resolve().parent
CHECKPOINT=Path("/media/scott/SSD_4TB/hf-cache/huggingface/hub/models--openbmb--MiniCPM-V-4.7-1B/snapshots/286817f1132b998913a129a221c89b4402b4ab80")
PROTOCOL_SHA="d9d95a879f3c12a5168c86e7f24969e2e6ce92a6adcfc55b92b7cb4d6f651bee"
EXP051_RUNNER_SHA="fdd90719644770e685aee84e82ea37b4add347723dce360458f6d668eb190977"
FIXTURE_SHA="af188e9d490d420e02cc67d83a3405d37061fd281012f2dfb67efb8864740f9c"
REVISION="286817f1132b998913a129a221c89b4402b4ab80"

def preflight(out):
    if sha_file(ROOT/"EXP052-PROTOCOL-2026-10-16.md")!=PROTOCOL_SHA:
        raise RuntimeError("HOLD preregistration drift")
    if sha_file(ROOT/"k2_qwen4b_exp051.py")!=EXP051_RUNNER_SHA:
        raise RuntimeError("HOLD imported EXP051 fixture/score drift")
    if out.exists() or out.parent.resolve()!=ROOT.resolve():
        raise RuntimeError("HOLD output must be new research file")
    if available_gib()<38:
        raise RuntimeError("HOLD insufficient CPU RAM")
    if not CHECKPOINT.is_dir() or CHECKPOINT.is_symlink():
        raise RuntimeError("HOLD missing/untrusted pinned checkpoint")
    for name in ("config.json","tokenizer.json","tokenizer_config.json","model.safetensors"):
        p=CHECKPOINT/name
        if not p.is_file() or p.stat().st_size<1000:
            raise RuntimeError("HOLD incomplete checkpoint "+name)
    cfg=json.loads((CHECKPOINT/"config.json").read_text())
    if (cfg.get("model_type"),cfg.get("text_config",{}).get("model_type"))!=("minicpmv4_7","qwen3_5_text"):
        raise RuntimeError("HOLD architecture not MiniCPM-V 4.7/Qwen3.5")
    if (CHECKPOINT/"model.safetensors").stat().st_size!=3109054480:
        raise RuntimeError("HOLD weight size mismatch")
    return {n:sha_file(CHECKPOINT/n) for n in ("config.json","tokenizer.json","tokenizer_config.json","model.safetensors")}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--execute",action="store_true")
    ap.add_argument("--out",type=Path)
    args=ap.parse_args()
    rows=make()
    receipt=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
    if len(rows)!=80 or receipt!=FIXTURE_SHA:
        raise RuntimeError("HOLD EXP051 fixture drift")
    print("EXP052_FROZEN_FIXTURE",len(rows),receipt,flush=True)
    if not args.execute:
        print("EXP052_DRY_RUN_NO_MODEL",flush=True);return
    if args.out is None:raise RuntimeError("HOLD explicit research output required")
    source_sha=preflight(args.out)
    import transformers
    from transformers import AutoTokenizer,MiniCPMV4_7ForConditionalGeneration
    if transformers.__version__!="5.20.1":raise RuntimeError("HOLD wrong transformer runtime")
    tok=AutoTokenizer.from_pretrained(str(CHECKPOINT),local_files_only=True,trust_remote_code=False)
    t0=time.monotonic()
    model=MiniCPMV4_7ForConditionalGeneration.from_pretrained(str(CHECKPOINT),
        local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,
        use_safetensors=True,low_cpu_mem_usage=True)
    model.eval().to("cpu").requires_grad_(False)
    print("EXP052_MODEL_LOADED",round(time.monotonic()-t0,2),flush=True)
    try:observations,elapsed=score(model,tok,rows)
    finally:
        del model,tok;gc.collect()
    if len(observations)!=80:raise RuntimeError("HOLD incomplete inference")
    result={"model":"minicpm-v4.7-1b","hf_revision":REVISION,
            "protocol_sha256":PROTOCOL_SHA,"exp051_fixture_sha256":receipt,
            "exp051_runner_sha256":EXP051_RUNNER_SHA,
            "sources_sha256":source_sha,"forward_passes":len(observations),
            "timed_forward_seconds":elapsed,"transformers_version":"5.20.1",
            "comparison_limit":"raw text continuation; instruction-tuned multimodal checkpoint, not human-evaluated",
            "summary":summarize(observations)}
    args.out.open("x").write(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("EXP052_COMPLETE",json.dumps(result["summary"],sort_keys=True),flush=True)
if __name__=="__main__":main()