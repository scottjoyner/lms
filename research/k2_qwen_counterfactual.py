#!/usr/bin/env python3
"""EXP032E: Same Qwen YES/NO judge for K2 candidate with source/question swaps.

Frozen models, deny-only results, zero raw-generation or human-gold export.
"""
import argparse
import gc
import hashlib
import os
import sys
import time
from collections import Counter
from pathlib import Path
for key,value in {'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','CUDA_VISIBLE_DEVICES':'-1',
                  'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}.items():os.environ[key]=value
import torch
import k2_qwen_candidate_guard as priorguard

PROTOCOL='b28b31a7f43a95bbdfb7755132fa09fe877121b309e4d2fbe87f58f961cb7f97'
PARENT='b6c04efdfd42187c7bcd8f470500467191d7ffe37ab4945cb4b37510510742fc'
CONDITIONS=('original','evidence_swap','question_swap')
EMPTY_EVIDENCE='0000 1111 2222 3333'
START=time.monotonic()
torch.set_num_threads(2)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def donors(rows,idx,candidate):
    current=rows[idx]
    possibilities=[r['context'] for r in rows
                   if r['article_index']!=current['article_index'] and
                   not priorguard.word_in(r['context'],candidate)]
    if possibilities:
        passage=min(possibilities,key=lambda t:hashlib.sha256(t.encode()).hexdigest())
        return passage,False
    assert not priorguard.word_in(EMPTY_EVIDENCE,candidate)
    return EMPTY_EVIDENCE,True

def counterfactuals(rows,candidates,selected):
    records=[];fallback=0
    by_article={r['article_index']:r['question'] for r in reversed(rows)}
    if set(by_article)!=set(range(8)):raise RuntimeError('Missing 8 original article questions')
    for i in selected:
        case=rows[i];candidate=candidates[i]['question_exclusion']
        swapped,fell_back=donors(rows,i,candidate)
        fallback+=int(fell_back)
        other_q=by_article[(case['article_index']+1)%8]
        choices={'original':(case['context'],case['question']),
                 'evidence_swap':(swapped,case['question']),
                 'question_swap':(case['context'],other_q)}
        for mode in CONDITIONS:
            context,question=choices[mode]
            records.append({'index':i,'mode':mode,'prompt':priorguard.verdict_prompt(context,question,candidate)})
    if len(records)!=3*len(selected):raise RuntimeError('Incomplete controlled 3-way prompt triplets')
    # Group by condition so original source prompts retain EXP032D's identical
    # two-candidate Qwen batching rather than mixing in swapped-length rows.
    records.sort(key=lambda row:CONDITIONS.index(row['mode']))
    return records,fallback

def summarize(paired,fallback,receipt):
    if set(paired)!=set(CONDITIONS):raise RuntimeError('Missing controlled conditions')
    if any(len(paired[p])!=11 for p in CONDITIONS):raise RuntimeError('Incomplete 11-per-condition responses')
    summary={mode:{name:sum(v==name for v in paired[mode].values())
                   for name in ('YES','NO','MALFORMED')} for mode in CONDITIONS}
    if summary['original']!={'YES':10,'NO':1,'MALFORMED':0}:
        raise RuntimeError('Original EXP032D Qwen judge outputs did not exactly reproduce')
    rows=['# EXP-032E — Frozen Qwen candidate judge under source and question swaps',
          '', '**2026-10-14 | actual paired counterfactual, research-only.**',
          'Original candidate, unchanged Qwen YES/NO question-specific prompt and',
          'same checkpoint. Original SQuAD2 human-authored public dev source',
          'Rajpurkar et al., 2018, CC BY-SA 4.0,',
          'https://rajpurkar.github.io/SQuAD-explorer/',
          '', '## Frozen custody','',
          f'Pre-inference EXP032E protocol SHA256 `{PROTOCOL}`.',
          f'Exact measured EXP032D Qwen/K2 ancestor Markdown SHA256 `{PARENT}`.',
          f'Human-annotated eight-article source selection SHA256 `{receipt}`.',
          f'K2 official checkpoint revision `{priorguard.K2_REV}`.',
          f'Qwen3.5-0.8B official checkpoint revision `{priorguard.QWEN_REV}`.',
          'K2 original 32-token native plain generation, Qwen official native',
          'chat verdict format with max 16 new tokens and exact YES/NO parser.',
          'Eleven candidates had passed the original lexical/question gates.',
          '', '## Paired Qwen counterfactual response inventory','',
          '| Evidence/question condition | Strict YES /11 | Strict NO /11 | Malformed/cap /11 |',
          '|---|---:|---:|---:|']
    for mode in CONDITIONS:
        vals=summary[mode]
        rows.append(f"| {mode} | {vals['YES']} | {vals['NO']} | {vals['MALFORMED']} |")
    for mode in CONDITIONS[1:]:
        yy=sum(paired['original'][i]=='YES' and paired[mode][i]=='YES' for i in paired['original'])
        yn=sum(paired['original'][i]=='YES' and paired[mode][i]=='NO' for i in paired['original'])
        ny=sum(paired['original'][i]=='NO' and paired[mode][i]=='YES' for i in paired['original'])
        rows.extend(['',f'Original-YES→{mode}-YES: **{yy}/10**.',
                         f'Original-YES→{mode}-NO: **{yn}/10**.',
                         f'Original-NO→{mode}-YES: **{ny}/1**.'])
    rows+=['',f'Donor passage lacked the source-visible candidate in all 11 cases;',
          f'fixed digits-only empty-evidence fallback count: **{fallback}/11**.',
          '', '## What the control establishes and does not establish','',
          'Removing the actual passage is a negative-control perturbation, not a',
          'general proof of lack of indirectly implied evidence. A swapped question',
          'is NOT a separately human-adjudicated negative and may coincidentally',
          'be semantically compatible with the proposed candidate. All 11',
          'candidates and prompts were chosen before viewing the new Qwen scores;',
          'the gold answer and human unanswerability flag were never given to the',
          'verifier or used to select donor passages. This is a public, previously',
          'exposed SQuAD2 subset with eight distinct article source groups.',
          'Do not conclude that Qwen is an independently validated entailment',
          'oracle, stop relying on an abstention gate, or promote K2/Qwen to',
          'production. All model weights, graph, NAS, router and live services',
          'were left unchanged. No raw human text/candidates/verdict strings',
          'were written beyond aggregate human-readable Markdown.',
          '', '## Replay','',
          '`python research/k2_qwen_counterfactual.py --evidence-dir /home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026 --out /home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026/EXPERIMENT-032E-COUNTERFACTUAL-JUDGE.md`','']
    return '\n'.join(rows),summary

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--preflight-only',action='store_true')
    args=p.parse_args()
    if args.out.exists():raise SystemExit('STOP: measured report already exists')
    if sha(args.evidence_dir/'EXPERIMENT-032E-COUNTERFACTUAL-JUDGE-PROTOCOL.md')!=PROTOCOL:
        raise RuntimeError('Pre-inference EXP032E preregistration mismatch')
    if sha(args.evidence_dir/'EXPERIMENT-032D-QWEN-VERIFIER.md')!=PARENT:
        raise RuntimeError('Original 032D measured Qwen ancestor mismatch')
    sys.path.insert(0,str(args.evidence_dir))
    import k2_human_grounded_qa as prior
    import run_exp021_free_response as loaders
    from transformers import AutoTokenizer
    import run_exp018_k2_intermediate_gates as e18
    import k2_posthoc_boundary as boundary
    tok=AutoTokenizer.from_pretrained(str(e18.stage_path('mopd_249')),
            local_files_only=True,trust_remote_code=False)
    rows,receipt=boundary.preflight(args.evidence_dir,prior,tok)
    rows=[r for r in rows if r['position']=='short']
    if len(rows)!=16:raise RuntimeError('Bad original short cohort')
    print('EXP032E_SOURCE_PREFLIGHT',receipt,'n',len(rows),'mem_gib',round(priorguard.memory_gib(),1),flush=True)
    if args.preflight_only:return
    if priorguard.memory_gib()<36:raise RuntimeError('Memory below 36GiB pre-K2')
    model,tok,k2_custody=loaders.load_model('mopd_249')
    if k2_custody['revision']!=priorguard.K2_REV:raise RuntimeError('Wrong K2 weights')
    tok.padding_side='left';candidates=[];control=[]
    try:
        for idx in range(0,len(rows),2):
            if priorguard.memory_gib()<22:raise RuntimeError('K2 batch RAM gate')
            sub=rows[idx:idx+2]
            tokens=tok([r['prompt'] for r in sub],padding=True,return_tensors='pt',truncation=False)
            tokens.pop('token_type_ids',None)
            with torch.inference_mode():
                out=model.generate(**tokens,max_new_tokens=32,do_sample=False,num_beams=1,
                        use_cache=True,eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id)
            for row,ids in zip(sub,out[:,tokens['input_ids'].shape[1]:]):
                vals=ids.tolist();end=next((i for i,v in enumerate(vals) if v==tok.eos_token_id),len(vals))
                txt=tok.decode(vals[:end],skip_special_tokens=True)
                control.append(prior.score_one(row,txt,end>=32))
                candidates.append(priorguard.candidate_rules(priorguard.candidate_from_text(txt),
                                                                row['context'],row['question']))
            del tokens,out
    finally:
        del model;gc.collect()
    if (len(candidates)!=16 or sum(s['lenient'] for s in control if not s['impossible'])!=7
            or sum(s['hit_cap'] for s in control)!=9 or sum(s['strict'] for s in control)!=0):
        raise RuntimeError('Original K2 generation did not reproduce')
    chosen=[i for i,c in enumerate(candidates) if c['verified_candidate_eligible']]
    if len(chosen)!=11:raise RuntimeError('Original 11 Qwen-eligible candidates not reproduced')
    prompts,fallback=counterfactuals(rows,candidates,chosen)
    if priorguard.memory_gib()<36:raise RuntimeError('Memory below 36GiB pre-Qwen')
    qwen,qtok,qwen_custody=loaders.load_model('qwen08_chat')
    if qwen_custody['revision']!=priorguard.QWEN_REV or not qwen_custody['native_chat_template']:
        raise RuntimeError('Wrong Qwen native checkpoint/chat template')
    qtok.padding_side='left';scores={mode:{} for mode in CONDITIONS}
    try:
        for idx in range(0,len(prompts),2):
            if priorguard.memory_gib()<22:raise RuntimeError('Qwen batch RAM gate')
            sub=prompts[idx:idx+2]
            x=[qtok.apply_chat_template([{'role':'user','content':r['prompt']}],
                     add_generation_prompt=True,tokenize=False) for r in sub]
            tokens=qtok(x,padding=True,return_tensors='pt',add_special_tokens=False,truncation=False)
            tokens.pop('token_type_ids',None)
            if tokens['input_ids'].shape[1]>1400:raise RuntimeError('Verifier input too long')
            with torch.inference_mode():
                out=qwen.generate(**tokens,do_sample=False,num_beams=1,
                       max_new_tokens=16,use_cache=True,
                       eos_token_id=qtok.eos_token_id,pad_token_id=qtok.pad_token_id)
            for row,ids in zip(sub,out[:,tokens['input_ids'].shape[1]:]):
                vals=ids.tolist();end=next((i for i,v in enumerate(vals) if v==qtok.eos_token_id),len(vals))
                decoded=qtok.decode(vals[:end],skip_special_tokens=True)
                scores[row['mode']][row['index']]=priorguard.normalized_verdict(decoded,end>=16)
            if idx%8==0:print('EXP032E_QWEN_PROGRESS',idx+len(sub),'/',len(prompts),flush=True)
            del tokens,out
    finally:
        del qwen;gc.collect()
    result,summary=summarize(scores,fallback,receipt)
    tmp=args.out.with_name(args.out.name+'.tmp')
    if tmp.exists():raise RuntimeError('Scratch report collision')
    tmp.write_text(result,encoding='utf8');tmp.replace(args.out)
    print('EXP032E_COMPLETE',sha(args.out),'scores',summary,'fallback',fallback,
          'elapsed_s',round(time.monotonic()-START,1),flush=True)

if __name__=='__main__':main()
