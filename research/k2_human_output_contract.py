#!/usr/bin/env python3
"""EXP030: SQuAD2 human-gold K2 format-only vs budget-only intervention.

Frozen checkpoint; exact EXP029 source and parent baseline pinned.
Outputs aggregate human-readable Markdown; never writes raw generations.
"""
import argparse
import gc
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

for key,value in {'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
                  'CUDA_VISIBLE_DEVICES':'-1','OPENBLAS_NUM_THREADS':'1',
                  'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}.items():os.environ[key]=value
import torch

PROTOCOL_SHA='c046e294c766385e9500e47cf971d4b8d7421683456e43bc29e2ad367f868727'
SOURCE_SHA='aad6e578544e2a3f100ab4b5c93d7ee6158fbfbe14bd426b8bffd9f9d1cde23b'
BASE_PROMPT_SHA='6513d65b456a0f52d230e386078d732a4b2800663c4d9b11c76b3cd569651c2a'
PARENT_SHA='0ab5ac2d744362969b97618b32ce9ad536ae32bda820a4bfac79410cc7c17c71'
MODEL_REV='7abae15437c2e7f1b2cd986be562344f204a17f3'
ARMS={'strict32':32,'budget64':64}
PLACEMENTS=('short','gap')
BASE_INSTRUCTION=('Answer the question using only the passage. '
    'Give the shortest answer from the passage. '
    'If the passage does not answer the question, respond exactly UNKNOWN.\n')
STRICT_INSTRUCTION=('Respond with exactly ONE WORD (or UNKNOWN). Do not explain. '
    'Using only the passage, write the shortest one-word answer, '
    'or exactly UNKNOWN when it cannot be answered.\n')
BATCH=2
START=time.monotonic()
torch.set_num_threads(2)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def memory_gib():
    return next(int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
                if line.startswith('MemAvailable:'))/1024**2

def prepare(prior,tokenizer,arm):
    source,src_sha,eligible=prior.source_rows()
    if src_sha!=SOURCE_SHA or eligible!=90:raise RuntimeError('Human SQuAD subset custody mismatch')
    original,original_sha,filler=prior.generate_rows(tokenizer,source,'mopd_249')
    if original_sha!=BASE_PROMPT_SHA or filler!=768:
        raise RuntimeError('Parent EXP029 exact prompt drift')
    out=[]
    for position in PLACEMENTS:
        selected=[row for row in original if row['position']==position]
        if len(selected)!=16:raise RuntimeError('Incomplete source position')
        for row in selected:
            prompt=row['prompt']
            if arm=='strict32':
                if prompt.count(BASE_INSTRUCTION)!=1 or not prompt.startswith(BASE_INSTRUCTION):
                    raise RuntimeError('Parent prompt was not exact; stop')
                prompt=STRICT_INSTRUCTION+prompt[len(BASE_INSTRUCTION):]
                if prompt[len(STRICT_INSTRUCTION):]!=row['prompt'][len(BASE_INSTRUCTION):]:
                    raise RuntimeError('Non-instruction source changed')
            elif prompt!=row['prompt']:
                raise RuntimeError('Budget-only source altered')
            out.append({**row,'prompt':prompt,
                        'prompt_tokens':len(tokenizer.encode(prompt,add_special_tokens=True))})
    if len(out)!=32 or len({(r['qa_id'],r['position']) for r in out})!=32:
        raise RuntimeError('Incomplete 32 case set')
    for i in range(16):
        if out[i]['qa_id']!=out[i+16]['qa_id']:raise RuntimeError('Question pairing lost')
        if abs(out[i+16]['prompt_tokens']-out[i]['prompt_tokens'])<650:
            raise RuntimeError('Gap not long enough')
    receipt=hashlib.sha256(json.dumps([{'qid':r['qa_id'],'placement':r['position'],
              'sha256':hashlib.sha256(r['prompt'].encode()).hexdigest(),
              'tokens':r['prompt_tokens']} for r in out],sort_keys=True,
              separators=(',',':')).encode()).hexdigest()
    return out,receipt

def scored(prior,row,output,cap):
    s=prior.score_one(row,output,cap)
    # Lexical-prefix diagnostic is explicit, not an acceptance substitute.
    first=re.match(r'\s*([A-Za-z][A-Za-z\-]*)',output)
    first=first.group(1).casefold() if first else ''
    golds={prior.norm(x) for x in row['answers']}
    s['first_gold']=int(not row['impossible'] and first in golds)
    s['first_unknown']=int(row['impossible'] and first=='unknown')
    return s

def metrics(scored_rows,pos):
    rows=[r for r in scored_rows if r['position']==pos]
    a=[r for r in rows if not r['impossible']]
    u=[r for r in rows if r['impossible']]
    if len(rows)!=16 or len(a)!=8 or len(u)!=8:raise RuntimeError('Missing one SQuAD source group')
    return {'strict':sum(x['strict'] for x in a),
            'lenient':sum(x['lenient'] for x in a),
            'first_gold':sum(x['first_gold'] for x in a),
            'unknown':sum(x['strict'] for x in u),
            'first_unknown':sum(x['first_unknown'] for x in u),
            'nonabstain':sum(x['nonabstain_impossible'] for x in u),
            'caps':sum(x['hit_cap'] for x in rows),
            'tokens':(min(x['tokens'] for x in rows),max(x['tokens'] for x in rows))}

def parent_baseline(directory):
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    import k2_human_qa_compare as compare
    if sha(directory/'EXPERIMENT-029-K2-HUMAN-QA.md')!=PARENT_SHA:
        raise RuntimeError('Original EXP029 K2 measured report changed')
    return compare.inputs(directory)['K2']

def render(arm,receipt,summary,baseline,source_receipt,model_receipt,tokens):
    out=['# EXP-030 — K2 human-gold output contract: '+arm,
        '', '**October 14, 2026 | actual frozen BF16 K2 generated-answer intervention.**',
        'Exactly eight human annotated answerable and eight unanswerable SQuAD 2.0',
        'dev items in two source placements; same checkpoint as EXP029.',
        '', '## Source and input custody','',
        f'EXP030 preregistered protocol SHA256 `{PROTOCOL_SHA}`.',
        f'Human-annotated SQuAD2 chosen-source SHA256 `{source_receipt}`.',
        f'Original EXP029 48-prompt tokenizer-specific SHA256 `{BASE_PROMPT_SHA}`.',
        f'Original EXP029 measured K2 result SHA256 `{PARENT_SHA}`.',
        f'Intervention actual 32-prompt/placement/token receipt `{receipt}`.',
        f"Model revision `{model_receipt['revision']}`; model config SHA256 `{model_receipt['model_config_sha256']}`.",
        f'Original K2 tokenizer SHA256 `{model_receipt["tokenizer_sha256"]}`.',
        'K2 was evaluated in plain completion format with no fabricated chat template.',
        f'Arm `{arm}`: maximum {tokens} new tokens, greedy, native EOS, two CPU threads.',
        '', '## Human-annotated output and UNKNOWN behavior','',
        '| Placement | Input tokens | Strict gold /8 | Gold somewhere /8 | First lexical word gold /8 | Strict UNKNOWN /8 | First lexical UNKNOWN /8 | Unsupported nonabstention /8 | Token caps /16 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for place in PLACEMENTS:
        x=summary[place]
        out.append(f"| {place} | {x['tokens'][0]}–{x['tokens'][1]} | {x['strict']} | {x['lenient']} | {x['first_gold']} | {x['unknown']} | {x['first_unknown']} | {x['nonabstain']} | {x['caps']} |")
    out+=['','## Descriptive deltas from separately measured EXP029 32-token baseline','',
          'Each comparison reuses the same eight articles. No significance inference.',
          '| Placement | Baseline strict /8 | New strict /8 | Baseline UNKNOWN /8 | New UNKNOWN /8 | Baseline caps /16 | New caps /16 |',
          '|---|---:|---:|---:|---:|---:|---:|']
    for place in PLACEMENTS:
        a=baseline[place];b=summary[place]
        out.append(f"| {place} | {a['exact']} | {b['strict']} | {a['unknown']} | {b['unknown']} | {a['caps']} | {b['caps']} |")
    out+=['','## Limits and scientific interpretation','',
          'The first lexical output word is a **diagnostic**, not a strict accepted',
          'answer. A correct gold word anywhere in verbose output is also lenient,',
          'not evidence that the whole response is correct. No source annotations',
          'or test outcomes were used to select the intervention prompts.',
          'The strict arm changes wording only; the budget arm changes only',
          'max-new-tokens. Neither alters K2 weights, YaRN, the SQuAD passages,',
          'or question identities. Comparisons use a prior separate model run.',
          'Unanswerable non-UNKNOWN is failure to abstain, not by itself proof',
          'of a fabricated fact. Generation token caps must be interpreted',
          'separately from correctness. Eight published-dev articles are too few',
          'for a generalized model ranking, and public SQuAD dev can be contaminated.',
          'Model training teacher, K2-to-Qwen layer parentage and production',
          'readiness are NOT inferable from this study.',
          'Original SQuAD2 development source by Rajpurkar et al. (2018), CC BY-SA 4.0.',
          'Source: https://rajpurkar.github.io/SQuAD-explorer/.',
          'No raw generations, answer text, full passages, prompts or tensor data',
          'are persisted; no NAS, graph, training, router or production mutation.',
          '', '## Reproduction','',
          'From pinned scottjoyner/lms research branch, with official BF16 model and',
          'SQuAD2 dev source already checksum-verified:', '', '```bash',
          'python research/k2_human_output_contract.py --arm '+arm+' --evidence-dir /home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026 --out /home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026/EXPERIMENT-030-'+arm.upper()+'.md',
          '```','']
    return '\n'.join(out)

def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--arm',required=True,choices=sorted(ARMS))
    cli.add_argument('--evidence-dir',type=Path,required=True)
    cli.add_argument('--out',type=Path,required=True)
    cli.add_argument('--preflight-only',action='store_true')
    args=cli.parse_args()
    if args.out.exists():raise SystemExit('STOP existing measured Markdown output')
    if sha(args.evidence_dir/'EXPERIMENT-030-PROTOCOL.md')!=PROTOCOL_SHA:
        raise RuntimeError('Preregistered EXP030 protocol changed')
    base=parent_baseline(args.evidence_dir)
    sys.path.insert(0,str(args.evidence_dir))
    import k2_human_grounded_qa as prior
    import run_exp021_free_response as backend
    from transformers import AutoTokenizer
    import run_exp018_k2_intermediate_gates as e18
    tok=AutoTokenizer.from_pretrained(str(e18.stage_path('mopd_249')),
                                       local_files_only=True,trust_remote_code=False)
    cases,receipt=prepare(prior,tok,args.arm)
    print('EXP030_SOURCE_PREFLIGHT',args.arm,'n',len(cases),'sha',receipt,
          'token_ranges',[(pos,min(r['prompt_tokens'] for r in cases if r['position']==pos),
                          max(r['prompt_tokens'] for r in cases if r['position']==pos))
                          for pos in PLACEMENTS],flush=True)
    if args.preflight_only:return
    if memory_gib()<36:raise RuntimeError('Initial memory below 36 GiB')
    model,tok,model_receipt=backend.load_model('mopd_249')
    if model_receipt['revision']!=MODEL_REV or model_receipt['native_chat_template']:
        raise RuntimeError('K2 model/original format identity mismatch')
    tok.padding_side='left'
    outputs=[]
    try:
        for start in range(0,len(cases),BATCH):
            if memory_gib()<22:raise RuntimeError('Memory below 22 GiB during model eval')
            group=cases[start:start+BATCH]
            x=tok([r['prompt'] for r in group],return_tensors='pt',padding=True,
                  add_special_tokens=True,truncation=False)
            x.pop('token_type_ids',None)
            if x['input_ids'].shape[1]>1300:raise RuntimeError('Unanticipated context length')
            with torch.inference_mode():
                generated=model.generate(**x,max_new_tokens=ARMS[args.arm],
                    do_sample=False,num_beams=1,use_cache=True,
                    pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
            new=generated[:,x['input_ids'].shape[1]:]
            for row,ids in zip(group,new):
                ids=ids.tolist();stops=[i for i,t in enumerate(ids) if t==tok.eos_token_id]
                stop=stops[0] if stops else len(ids)
                decoded=tok.decode(ids[:stop],skip_special_tokens=True)
                outputs.append(scored(prior,row,decoded,not stops and stop>=ARMS[args.arm]))
            if start%8==0:
                print('EXP030_PROGRESS',args.arm,start+len(group),'/',len(cases),
                      'elapsed_sec',round(time.monotonic()-START,1),flush=True)
            del x,generated,new
    finally:
        del model;gc.collect()
    if len(outputs)!=32:raise RuntimeError('Incomplete 32 generated model outputs')
    summary={pos:metrics(outputs,pos) for pos in PLACEMENTS}
    text=render(args.arm,receipt,summary,base,SOURCE_SHA,model_receipt,ARMS[args.arm])
    if not args.out.parent.is_dir():raise RuntimeError('Output directory not present')
    tmp=args.out.with_name(args.out.name+'.tmp')
    if tmp.exists():raise RuntimeError('Scratch file collision')
    tmp.write_text(text,encoding='utf8');tmp.replace(args.out)
    print('EXP030_PASS',args.arm,'sha',sha(args.out),'elapsed_s',round(time.monotonic()-START,1),
          'scores',summary,flush=True)

if __name__=='__main__':main()
