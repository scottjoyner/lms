#!/usr/bin/env python3
"""EXP032C frozen K2 actual posthoc EOS/alternate marker order witness."""
import argparse
import hashlib
import os
import sys
import time
from pathlib import Path
for k,v in {'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','CUDA_VISIBLE_DEVICES':'-1',
            'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}.items():os.environ[k]=v
import torch
PROTOCOL='2e72703b574f3fc4275578f701fa99b7ec377aa7c39e5ffa906f7b8fa25ad999'
EXP032B_SHA='a4c23ec645a5dfe825257ce8ed481d068011c8ecc0fbb5c07597075364ba8a4c'
MODEL_REV='7abae15437c2e7f1b2cd986be562344f204a17f3'
EOS=1
ALT=64019
MAX_NEW=32
START=time.monotonic()
torch.set_num_threads(2)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def memory():
    return next(int(x.split()[1]) for x in Path('/proc/meminfo').read_text().splitlines()
                if x.startswith('MemAvailable:'))/1024**2

def timeline(tokens):
    eos=next((i for i,t in enumerate(tokens) if t==EOS),None)
    end=eos+1 if eos is not None else len(tokens)
    # Generated PAD after the first EOS is not a model continuation.
    emitted=tokens[:end]
    alt=next((i for i,t in enumerate(emitted) if t==ALT),None)
    if eos is None and alt is None:cell='neither'
    elif eos is None:cell='alt_only'
    elif alt is None:cell='eos_only'
    else:cell='both'
    if cell=='both':
        order='alt_before_eos' if alt<eos else 'eos_before_alt'
        delta=eos-alt
        adjacency=('1' if delta==1 else '2' if delta==2 else '3' if delta==3 else '>3') if delta>0 else 'invalid'
    else:
        order=None;adjacency=None
    return {'cell':cell,'order':order,'adjacency':adjacency,'eos':eos,
            'alt':alt,'cap':int(eos is None and len(tokens)>=MAX_NEW),
            'effective_end':end}

def analyze(ids,tok,row,prior):
    t=timeline(ids)
    eos=t['eos'] if t['eos'] is not None else len(ids)
    boundary=min(eos,t['alt'] if t['alt'] is not None else eos)
    native=tok.decode(ids[:eos],skip_special_tokens=True)
    alt=tok.decode(ids[:boundary],skip_special_tokens=True)
    a=prior.score_one(row,native,bool(t['cap']))
    b=prior.score_one(row,alt,bool(t['cap']))
    return {'timeline':t,'changed':int(native!=alt),
            'impossible':row['impossible'],
            'native_strict':a['strict'],'alt_strict':b['strict'],
            'native_lenient':a['lenient'],'alt_lenient':b['lenient']}

def render(rows,receipt):
    count={key:sum(r['timeline']['cell']==key for r in rows)
           for key in ('both','alt_only','eos_only','neither')}
    order={key:sum(r['timeline']['order']==key for r in rows)
           for key in ('alt_before_eos','eos_before_alt')}
    adjacent={key:sum(r['timeline']['adjacency']==key for r in rows)
              for key in ('1','2','3','>3')}
    n_cap=sum(r['timeline']['cap'] for r in rows)
    n_changed=sum(r['changed'] for r in rows)
    def scores(key,impossible):return sum(r[key] for r in rows if r['impossible']==impossible)
    if sum(count.values())!=16 or len(rows)!=16:raise RuntimeError('Incomplete paired witness')
    # An exact original seed/runtime replay is an acceptance gate.
    if not (scores('native_strict',False)==0 and scores('native_strict',True)==0
            and scores('native_lenient',False)==7 and n_cap==9):
        raise RuntimeError('EXP029 original native source/generation control mismatched')
    lines=['# EXP-032C — Native K2 alternate boundary and EOS co-occurrence',
      '', '**2026-10-14 | actual frozen 16-short-prompt CPU generation witness.**',
      'Eight previously published human-annotated answerable and eight',
      'unanswerable SQuAD2 dev questions. No raw token trace or text retained.',
      'SQuAD2 authors Rajpurkar et al., 2018, CC BY-SA 4.0;',
      'https://rajpurkar.github.io/SQuAD-explorer/',
      '', '## Exact custody and unchanged EOS behavior','',
      f'Pre-inference EXP032C protocol SHA256 `{PROTOCOL}`.',
      f'EXP032B native/lexical-gate measured report SHA256 `{EXP032B_SHA}`.',
      f'Original human gold 8-article source receipt `{receipt["source"]}`.',
      f'Original K2 tokenizer/placement receipt `{receipt["prompts"]}`.',
      f'Official frozen K2 mOPD revision `{receipt["revision"]}`.',
      f'Config SHA256 `{receipt["config"]}`; tokenizer SHA256 `{receipt["tokenizer"]}`.',
      'Greedy 32 new tokens, unchanged native EOS ID1, alternate marker ID64019.',
      '', '## Temporal 2×2 native EOS versus alternate marker','',
      '| EOS and alternate marker event | Real generated traces /16 |',
      '|---|---:|']
    for key in ('both','alt_only','eos_only','neither'):
        lines.append(f'| {key.replace("_"," ")} | {count[key]} |')
    lines+=['','| Where both markers occur | Sequences |', '|---|---:|']
    for key in ('alt_before_eos','eos_before_alt'):
        lines.append(f'| {key.replace("_"," ")} | {order[key]} |')
    lines+=['','| EOS index minus alternate index, when alternate comes first | Count |',
            '|---|---:|']
    for key in ('1','2','3','>3'):
        lines.append(f'| {key} | {adjacent[key]} |')
    lines+=['',f'Native EOS missing after 32 generated tokens: **{n_cap}/16**.',
            f'Post-hoc alternate cut changed decoded string: **{n_changed}/16**.',
            '', '## Strict human-gold readout sanity controls','',
            '| Post-hoc decoder | Exact human gold /8 answerable | Strict UNKNOWN /8 negative | Gold word anywhere /8 |',
            '|---|---:|---:|---:|',
            f"| Native EOS | {scores('native_strict',False)} | {scores('native_strict',True)} | {scores('native_lenient',False)} |",
            f"| Earlier ID64019 or EOS | {scores('alt_strict',False)} | {scores('alt_strict',True)} | {scores('alt_lenient',False)} |",
            '', '## Boundary and safety implications','',
            'A generated alt marker immediately followed by native EOS is',
            'NOT evidence that a missed stop marker caused a long continuation.',
            'These markers were only observed under one tiny, public, already',
            'exposed eight-article SQuAD2 dev subset. A co-occurrence count',
            'does NOT confer normative end-of-answer semantics on ID64019.',
            'Do not alter production stop lists, suppress model reasoning,',
            'decrease monitoring, or accept lexical source matching as semantic',
            'evidence verification. Grounded K2 answer/UNKNOWN remains NO-GO.',
            'All model weights, router, NAS, graph and production services unchanged.',
            '', '## Reproducibility','',
            '`python research/k2_stop_pairing.py --evidence-dir /home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026 --out /home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026/EXPERIMENT-032C-STOP-PAIRING.md`', '']
    return '\n'.join(lines)

def main():
    c=argparse.ArgumentParser(description=__doc__)
    c.add_argument('--evidence-dir',type=Path,required=True)
    c.add_argument('--out',type=Path,required=True)
    c.add_argument('--preflight-only',action='store_true')
    a=c.parse_args()
    if a.out.exists():raise SystemExit('STOP destination exists')
    if sha(a.evidence_dir/'EXPERIMENT-032C-STOP-PAIRING-PROTOCOL.md')!=PROTOCOL:
        raise RuntimeError('EXP032C preregistration drift')
    if sha(a.evidence_dir/'EXPERIMENT-032B-STOP-POSTHOC.md')!=EXP032B_SHA:
        raise RuntimeError('Source EXP032B witness changed')
    sys.path.insert(0,str(a.evidence_dir))
    import k2_human_grounded_qa as prior
    import run_exp021_free_response as backend
    import run_exp018_k2_intermediate_gates as exp18
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(exp18.stage_path('mopd_249')),
                                       local_files_only=True,trust_remote_code=False)
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    import k2_posthoc_boundary as earlier
    original,source=earlier.preflight(a.evidence_dir,prior,tok)
    cases=[r for r in original if r['position']=='short']
    if len(cases)!=16 or sum(r['impossible'] for r in cases)!=8:
        raise RuntimeError('Wrong 16-prompt cohort')
    print('EXP032C_PREFLIGHT',source,'n',len(cases),'RAM_GiB',round(memory(),2),flush=True)
    if a.preflight_only:return
    if memory()<36:raise RuntimeError('Initial admission <36GiB')
    model,tok,receipt=backend.load_model('mopd_249')
    try:
        if receipt['revision']!=MODEL_REV or model.config.eos_token_id!=EOS:
            raise RuntimeError('Native K2 model/eos mismatch')
        tok.padding_side='left';observed=[]
        for i in range(0,len(cases),2):
            if memory()<22:raise RuntimeError('Per-batch memory stop')
            batch=cases[i:i+2]
            tokens=tok([r['prompt'] for r in batch],padding=True,
                       return_tensors='pt',truncation=False)
            tokens.pop('token_type_ids',None)
            with torch.inference_mode():
                gen=model.generate(**tokens,max_new_tokens=MAX_NEW,
                    do_sample=False,num_beams=1,use_cache=True,
                    eos_token_id=EOS,pad_token_id=tok.pad_token_id)
            for row,ids in zip(batch,gen[:,tokens['input_ids'].shape[1]:]):
                observed.append(analyze(ids.tolist(),tok,row,prior))
            if i%4==0:
                print('EXP032C_PROGRESS',i+len(batch),'/',len(cases),
                      'elapsed_sec',round(time.monotonic()-START,1),flush=True)
            del tokens,gen
        evidence={'source':source,'prompts':earlier.EXP029_PROMPTS,
                  'revision':receipt['revision'],
                  'config':receipt['model_config_sha256'],
                  'tokenizer':receipt['tokenizer_sha256']}
        doc=render(observed,evidence)
        tmp=a.out.with_name(a.out.name+'.tmp')
        if tmp.exists():raise RuntimeError('Scratch collision')
        tmp.write_text(doc,encoding='utf8');tmp.replace(a.out)
        print('EXP032C_COMPLETE',sha(a.out),'elapsed_sec',round(time.monotonic()-START,1))
    finally:
        del model
if __name__=='__main__':main()
