#!/usr/bin/env python3
"""EXP028: same-length prefix-vs-gap filler under pinned YaRN amplitude.

Measures whether *frequencies* matter when fact-answer distance grows.
Output observations directly to central knowledge Markdown, never JSON.
"""
import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import torch
import numpy as np
from transformers import AutoConfig,AutoTokenizer
from k2_yarn_components import (setup_rope,ram_gib,weight_receipts,sha,
                                 EXP018_SHA,PRIOR_WEIGHT_SHA)
import k2_rope_components_fixture as fixture

GROUPS=(0,8)
TARGET_TOKENS=(128,384)
PLACEMENTS=('prefix','gap')
FREQUENCIES=('default','yarn')
SOURCE_SHA='ef0f99caa44a96f689c4b80ebe8f842032289238513d99bfbfb44dfa3ca79523'
DERIVED_SHA='1565a9edb3e5829b767ffd750583e11f3dfce20e41642562829dd22b9ee8ca1b'
PROTOCOL_SHA='a65228e95ab5c5e06a78446a8228a8a80b482d0c83e66e9d592118793bcc745b'
BATCH=8

def filler_text(tok,target):
    # No surname/family tokens; identical filler string within placement pair.
    text=''.join(f'Ledger number {i:04d} records a gray tile beside a white cup.\n'
                 for i in range(160))
    ids=tok.encode(text,add_special_tokens=False)
    if len(ids)<target+50:raise RuntimeError('Insufficient filler token inventory')
    selected=tok.decode(ids[:target],skip_special_tokens=True)
    actual=len(tok.encode(selected,add_special_tokens=False))
    if abs(actual-target)>3:raise RuntimeError('Filler length drift')
    if any(name.lower() in selected.lower().split() for name in fixture.NAMES):
        raise RuntimeError('Name contamination in filler')
    return selected

def extend(row,filler,placement):
    first,tail=row['prompt'].split('\nStatement: ',1)
    fact,last=tail.split('\nQuestion: ',1)
    if placement=='prefix':
        prompt=f'{first}\n{filler}\nStatement: {fact}\nQuestion: {last}'
    elif placement=='gap':
        prompt=f'{first}\nStatement: {fact}\n{filler}\nQuestion: {last}'
    else:raise ValueError('Invalid filler placement')
    return prompt

def build(tok):
    fixture.check_disjoint()
    base=fixture.make()
    if fixture.receipt(base)['sha256']!=SOURCE_SHA:
        raise RuntimeError('Parent EXP027 fixture digest mismatch')
    chosen=[row for row in base if row['group'] in GROUPS]
    if len(chosen)!=32:raise RuntimeError('Wrong four-source-group subset')
    records=[]
    for target in TARGET_TOKENS:
        fill=filler_text(tok,target)
        for placement in PLACEMENTS:
            for row in chosen:
                prompt=extend(row,fill,placement)
                n=len(tok.encode(prompt,add_special_tokens=False))
                loc_s=len(tok.encode(prompt[:prompt.index('\nStatement: ')],add_special_tokens=False))
                loc_q=len(tok.encode(prompt[:prompt.index('\nQuestion: ')],add_special_tokens=False))
                records.append({**row,'prompt':prompt,'target':target,
                                'placement':placement,'tokens':n,
                                'gap_tokens':loc_q-loc_s})
    if len(records)!=128:raise RuntimeError('Expected 256 distinct context rows')
    seen={}
    for row in records:
        key=(row['target'],row['group'],row['state'],row['role'],row['order'],row['condition'])
        seen.setdefault(key,{})[row['placement']]=row
    if len(seen)!=64:raise RuntimeError('Context paired source grouping error')
    for pair in seen.values():
        if set(pair)!=set(PLACEMENTS):raise RuntimeError('Incomplete placement pair')
        if abs(pair['prefix']['tokens']-pair['gap']['tokens'])>4:
            raise RuntimeError('Prompt lengths not controlled within 4 tokens')
        if pair['prefix']['gap_tokens']>=pair['gap']['gap_tokens']-80:
            raise RuntimeError('Source-to-question gap not meaningfully changed')
    receipt=hashlib.sha256(json.dumps(records,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return records,receipt

def predict(core,tok,head,rows):
    scored={}
    tok.padding_side='right'
    for offset in range(0,len(rows),BATCH):
        if ram_gib()<22:raise RuntimeError('RAM dropped below 22GiB')
        block=rows[offset:offset+BATCH]
        batch=tok([r['prompt'] for r in block],padding=True,return_tensors='pt',truncation=False)
        batch.pop('token_type_ids',None)
        if batch['input_ids'].shape[1]>650:raise RuntimeError('Unanticipated token length')
        pos=batch['attention_mask'].sum(1)-1
        with torch.inference_mode():
            x=core(**batch,use_cache=False).last_hidden_state
            logits=x[torch.arange(len(block)),pos].float()@head.T
            logits=(logits[:,0]-logits[:,1]).cpu().numpy()
        for row,v in zip(block,logits):
            key=(row['target'],row['placement'],row['group'],row['state'],row['role'],row['condition'],row['order'])
            if key in scored:raise RuntimeError('Duplicate observation')
            margin=float(v) if row['correct']=='A' else -float(v)
            scored[key]={'margin':margin,'raw_A':int(v>0),'correct':int(margin>0)}
        del batch,x,logits
    if len(scored)!=len(rows):raise RuntimeError('Incomplete observed conditions')
    return scored

def metric(scored,target,placement):
    eps=[]
    for g in GROUPS:
        for state in (0,1):
            for role in ('actor','second_party'):
                factual=[scored[target,placement,g,state,role,'factual',order] for order in (0,1)]
                neutral=[scored[target,placement,g,state,role,'neutral',order] for order in (0,1)]
                f=np.mean([r['margin'] for r in factual]); n=np.mean([r['margin'] for r in neutral])
                eps.append({'group':g,'role':role,'raw':sum(r['correct'] for r in factual),
                            'both':int(all(r['correct'] for r in factual)),
                            'debiased':int(f>0),'positive_fact':int(f>n),
                            'neutral_A':sum(r['raw_A'] for r in neutral)})
    if len(eps)!=8:raise RuntimeError('Wrong role-state count')
    totals={key:sum(row[key] for row in eps) for key in ('raw','both','debiased','positive_fact','neutral_A')}
    totals['roles']={role:{k:sum(row[k] for row in eps if row['role']==role)
                                for k in ('raw','both','debiased','positive_fact')}
                     for role in ('actor','second_party')}
    return totals

def report(receipt,records,stats,revision,freq_changed):
    lines=['# EXP-028 — Context gap versus length: does K2 YaRN frequency matter?',
           '', '**2026-10-14 | measured BF16 forward controls, research-only.**',
           'Same K2 midtraining-2 learned weights for both configurations,',
           'YaRN amplitude fixed at 1.277258872 and only rotary frequency swapped.',
           'Source experiment EXP027 had already been inspected: this is an',
           'exploratory position manipulation, not independently held-out gold.',
           '', '## Exact audit trail','',
           f'EXP027 fixture SHA256 `{SOURCE_SHA}`, two frozen name-group IDs {GROUPS}.',
           f'EXP028 derived-token-prompt digest `{receipt}`.',
           f'Pre-inference protocol SHA256 `{PROTOCOL_SHA}`.',
           f'EXP018 model evidence SHA256 `{EXP018_SHA}`.',
           f'EXP025 all-tensor ancestry SHA256 `{PRIOR_WEIGHT_SHA}`.',
           f'K2 frozen mid2 HF revision `{revision}`.',
           f'YaRN changes {freq_changed} inverse frequency vector elements.',
           'Filler targets 128/384 tokenized tokens. Placement A is before',
           'the fact, placement B is between fact and question; same filler',
           'text, same fact, same question and same answer options.',
           'Tokens/relative source gap were verified per prompt to within four',
           'tokens of matched paired total length. No material exported from model.',
           '', '## Measured exact position and prompt-length ranges','',
           '| Filler target | Placement | Prompt token range | Fact→question token gap range |',
           '|---:|---|---:|---:|']
    for target in TARGET_TOKENS:
        for placement in PLACEMENTS:
            rows=[r for r in records if r['target']==target and r['placement']==placement]
            lines.append(f"| {target} | {placement} | {min(r['tokens'] for r in rows)}–{max(r['tokens'] for r in rows)} | {min(r['gap_tokens'] for r in rows)}–{max(r['gap_tokens'] for r in rows)} |")
    lines+=['','## Frozen-YaRN-amplitude, frequency-only intervention','',
      'Per cell: 16 factual role/option questions, eight role-state episodes,',
      '16 neutral controls. `Both` means correct under both option orders.',
      '| Filler | Placement | Frequency | Raw factual /16 | Both orders /8 | Debiased /8 | Fact-reference /8 | Neutral-A /16 |',
      '|---:|---|---|---:|---:|---:|---:|---:|']
    for target in TARGET_TOKENS:
        for place in PLACEMENTS:
            for freq in FREQUENCIES:
                x=stats[freq,target,place]
                lines.append(f"| {target} | {place} | {freq} | {x['raw']} | {x['both']} | {x['debiased']} | {x['positive_fact']} | {x['neutral_A']} |")
    lines+=['','## Controlled paired frequency contrasts','',
       'All deltas are YaRN-frequency minus default-frequency with identical',
       'scaling, texts, answer choices and learned weights. Sample is extremely',
       'small: two name groups, 16 factual answers per cell. No bootstrap',
       'or population significance claims are justified.',
       '| Filler | Placement | Raw correct delta | Both-order delta | Debiased delta | Positive fact delta |',
       '|---:|---|---:|---:|---:|---:|']
    for target in TARGET_TOKENS:
        for place in PLACEMENTS:
            a=stats['default',target,place];b=stats['yarn',target,place]
            lines.append(f"| {target} | {place} | {b['raw']-a['raw']:+d} | {b['both']-a['both']:+d} | {b['debiased']-a['debiased']:+d} | {b['positive_fact']-a['positive_fact']:+d} |")
    lines+=['','## Interpretation limits','',
      'The same-length prefix-vs-gap perturbation tests whether differences',
      'arise from total prompt length or specific source-question separation.',
      'It does not eliminate distraction, recency, lexical and synthetic-label',
      'effects. Raw choice logits are not free-form generation.',
      'The historical EXP027 short-prompt null frequency result remains',
      'valid only on that short-prompt fixture.',
      'No claim about actual 128K-token capability, named training teacher,',
      'Qwen3.5 parentage, generalized semantic correctness, or production',
      'readiness follows from two selected author-generated name groups.',
      'The next strict gate is a larger external adjudicated long-context',
      'corpus, multiple independently varied source distances, and full-name',
      'generation with explicit abstention.',
      '', '## Reproduce from pinned source','', '```bash',
      'python research/k2_context_gap.py --evidence-dir /home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026 --out /home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026/EXPERIMENT-028-CONTEXT-GAP.md',
      '```','']
    return '\n'.join(lines)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--preflight-only',action='store_true')
    args=p.parse_args()
    if args.out.exists():raise SystemExit('STOP: output already exists')
    if not args.preflight_only and ram_gib()<38:raise RuntimeError('Initial RAM admission below 38GiB')
    if sha(args.evidence_dir/'EXPERIMENT-028-PROTOCOL.md')!=PROTOCOL_SHA:raise RuntimeError('Preregistered protocol SHA mismatch')
    if sha(args.evidence_dir/'results_exp018_intermediate_k2_20261014.json')!=EXP018_SHA:raise RuntimeError('Model custody mismatch')
    if sha(args.evidence_dir/'EXPERIMENT-025-FULL-WEIGHT-DRIFT.md')!=PRIOR_WEIGHT_SHA:raise RuntimeError('Full weight evidence mismatch')
    sys.path.insert(0,str(args.evidence_dir))
    import exp015_fixture
    import run_exp018_k2_intermediate_gates as e18
    import run_exp016_k2_pretrain_release as e16
    import run_exp015_role_qa as qa
    base=AutoTokenizer.from_pretrained(str(e16.BASE),local_files_only=True,trust_remote_code=False)
    records,receipt=build(base)
    if receipt!=DERIVED_SHA:raise RuntimeError('Matched-position derived prompt digest drift')
    print('EXP028_PROMPT_SOURCE_RECEIPT',receipt,'n',len(records),flush=True)
    if args.preflight_only:return
    model,tok,parity=e18.load_stage('mid2_47684',base,exp015_fixture.make())
    if parity!=256 or sum(tok.encode(r['prompt'],add_special_tokens=False)==base.encode(r['prompt'],add_special_tokens=False) for r in records)!=len(records):
        raise RuntimeError('Tokenizer parity drift')
    core=model.model;before=weight_receipts(core);orig=core.rotary_emb
    cfg=AutoConfig.from_pretrained(str(e18.stage_path('rl_merged')),local_files_only=True,trust_remote_code=True)
    yarn=type(orig)(config=cfg,device=torch.device('cpu'))
    if orig.rope_type!='default' or cfg.rope_parameters['rope_type']!='yarn' or float(yarn.attention_scaling)<=1.2:
        raise RuntimeError('Positional control provenance')
    head=model.lm_head.weight[qa.token_ids(tok),:].float().clone()
    outcomes={}
    try:
        for freq in FREQUENCIES:
            core.rotary_emb=setup_rope(orig,yarn,freq,'yarn')
            for target in TARGET_TOKENS:
                for place in PLACEMENTS:
                    rows=[r for r in records if r['target']==target and r['placement']==place]
                    x=predict(core,tok,head,rows)
                    outcomes[freq,target,place]=metric(x,target,place)
                    print('EXP028_CELL',freq,target,place,outcomes[freq,target,place],flush=True)
                    del x;gc.collect()
    finally:
        core.rotary_emb=orig
    if weight_receipts(core)!=before:raise RuntimeError('Frozen tensor signature changed')
    if len(outcomes)!=8:raise RuntimeError('Incomplete eight-cell controlled result')
    ref=json.loads((args.evidence_dir/'results_exp018_intermediate_k2_20261014.json').read_text())
    revision=ref['intermediate_config_and_index_receipts']['mid2_47684']['revision']
    doc=report(receipt,records,outcomes,revision,int(torch.count_nonzero(orig.inv_freq!=yarn.inv_freq)))
    if not args.out.parent.is_dir():raise RuntimeError('Output directory absent')
    tmp=args.out.with_name(args.out.name+'.tmp')
    if tmp.exists():raise RuntimeError('Scratch file collision')
    tmp.write_text(doc,encoding='utf8');tmp.replace(args.out)
    print('EXP028_COMPLETE',sha(args.out),len(doc.splitlines()),flush=True)

if __name__=='__main__':main()
