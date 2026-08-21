#!/usr/bin/env python3
"""Dependency-light OpenAI-compatible llama-server latency/concurrency probe."""
from __future__ import annotations
import argparse, concurrent.futures, json, statistics, time
from dataclasses import dataclass, asdict
from typing import Any
import requests

@dataclass
class Sample:
    request_id:int; ok:bool; status_code:int|None; ttft_s:float|None; wall_s:float
    prompt_tokens:int|None; completion_tokens:int|None; decode_tps:float|None; error:str|None=None

def make_prompt(repetitions:int, nonce:str)->str:
    seed=("Benchmark this deterministic local-inference workload. Preserve all details, reason carefully, "
          "and return a concise technical summary. ROCm llama.cpp context batching KV cache speculative decoding concurrency. ")
    return (seed*repetitions).strip() + (f"\nnonce={nonce}" if nonce else "")

def run_one(args:argparse.Namespace, request_id:int)->Sample:
    url=args.endpoint.rstrip('/')+'/chat/completions'
    nonce='' if args.cache_mode=='warm' else f'{args.label}-{request_id}-{time.time_ns()}'
    payload:dict[str,Any]={"model":args.model,"messages":[{"role":"user","content":make_prompt(args.prompt_repetitions,nonce)}],
        "max_tokens":args.max_tokens,"temperature":args.temperature,"stream":True,"stream_options":{"include_usage":True}}
    headers={"Content-Type":"application/json"}
    if args.api_key: headers["Authorization"]=f"Bearer {args.api_key}"
    started=time.perf_counter(); first=None; usage={}; status=None
    try:
        with requests.post(url,json=payload,headers=headers,stream=True,timeout=args.timeout) as resp:
            status=resp.status_code; resp.raise_for_status()
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith('data:'): continue
                data=raw[5:].strip()
                if data=='[DONE]': break
                try: obj=json.loads(data)
                except json.JSONDecodeError: continue
                if obj.get('usage'): usage=obj['usage']
                choices=obj.get('choices') or []
                if choices:
                    delta=choices[0].get('delta') or {}
                    if (delta.get('content') or delta.get('reasoning_content')) and first is None: first=time.perf_counter()
        ended=time.perf_counter(); wall=ended-started; ttft=None if first is None else first-started
        pt=usage.get('prompt_tokens'); ct=usage.get('completion_tokens'); dt=None
        if ct and first is not None and ended>first: dt=ct/(ended-first)
        return Sample(request_id,True,status,ttft,wall,pt,ct,dt)
    except Exception as exc:
        return Sample(request_id,False,status,None,time.perf_counter()-started,None,None,None,repr(exc))

def quantile(values:list[float], q:float)->float|None:
    if not values:return None
    values=sorted(values); idx=min(len(values)-1,max(0,round((len(values)-1)*q))); return values[idx]

def statblock(vals:list[float])->dict[str,float|None]:
    return {"mean":statistics.fmean(vals) if vals else None,"p50":quantile(vals,.50),"p95":quantile(vals,.95),"p99":quantile(vals,.99),"max":max(vals) if vals else None}

def main()->int:
    p=argparse.ArgumentParser();p.add_argument('--endpoint',default='http://127.0.0.1:8080/v1');p.add_argument('--model',default='local-model')
    p.add_argument('--concurrency',type=int,default=1);p.add_argument('--requests',type=int,default=5);p.add_argument('--prompt-repetitions',type=int,default=256)
    p.add_argument('--max-tokens',type=int,default=256);p.add_argument('--temperature',type=float,default=0.0);p.add_argument('--timeout',type=float,default=600.0)
    p.add_argument('--api-key',default='');p.add_argument('--label',default='');p.add_argument('--cache-mode',choices=['warm','cold'],default='warm');p.add_argument('--output',required=True);args=p.parse_args()
    started=time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        samples=list(pool.map(lambda i:run_one(args,i),range(args.requests)))
    elapsed=time.perf_counter()-started; ok=[s for s in samples if s.ok]; tt=[s.ttft_s for s in ok if s.ttft_s is not None]; dw=[s.wall_s for s in ok]; dec=[s.decode_tps for s in ok if s.decode_tps is not None]
    out_tokens=sum(s.completion_tokens or 0 for s in ok)
    result={"label":args.label,"cache_mode":args.cache_mode,"endpoint":args.endpoint,"model":args.model,"concurrency":args.concurrency,"requests":args.requests,
      "successful_requests":len(ok),"failed_requests":len(samples)-len(ok),"elapsed_s":elapsed,"output_tokens":out_tokens,
      "aggregate_output_tps":out_tokens/elapsed if elapsed>0 else None,"ttft_s":statblock(tt),"wall_s":statblock(dw),"per_request_decode_tps":statblock(dec),"samples":[asdict(s) for s in samples]}
    with open(args.output,'w',encoding='utf-8') as fh: json.dump(result,fh,indent=2,sort_keys=True);fh.write('\n')
    print(json.dumps(result,sort_keys=True));return 0 if len(ok)==len(samples) else 1
if __name__=='__main__':raise SystemExit(main())
