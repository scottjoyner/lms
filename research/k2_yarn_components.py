#!/usr/bin/env python3
"""EXP027: K2 YaRN frequency versus rotary amplitude controlled decomposition.

Frozen 255-tensor K2 mid2 weights. Four rotary-module conditions in RAM.
New synthetic groups, A/B order pairs and neutral fact ablations.
Write observations directly to Markdown in central knowledge.
"""
import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path

for k,v in {'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','CUDA_VISIBLE_DEVICES':'-1',
            'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}.items():os.environ[k]=v
import numpy as np
import torch
from transformers import AutoTokenizer,AutoConfig
from k2_rope_factorial import weight_receipts
import k2_rope_components_fixture as fixture

FIXTURE_SHA='ef0f99caa44a96f689c4b80ebe8f842032289238513d99bfbfb44dfa3ca79523'
EXP018_SHA='3818a421a9b9cbe6f52b0a27d8c3da69be0ee1bd4d50071c190d50ba95ee893a'
PRIOR_WEIGHT_SHA='169507cd918359b189abf208c5708af5a301a8017096b23642864f455f86821a'
PROTOCOL_SHA='7b6bc39d5a674eb8a5eabb772371abd7d2130df84830a418ce10df29ad1737b5'
CORNERS=(('default','default'),('yarn','default'),('default','yarn'),('yarn','yarn'))
BATCH=8
METRICS=('raw_correct','both_correct','debiased_correct','positive_fact_signal')
torch.set_num_threads(2)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def ram_gib():
    free=next(int(x.split()[1]) for x in Path('/proc/meminfo').read_text().splitlines()
              if x.startswith('MemAvailable:'))
    return free/1024**2

def setup_rope(original,yarn,frequency,amplitude):
    rot=type(original)(config=original.config,device=torch.device('cpu'))
    src=original if frequency=='default' else yarn
    scale=original if amplitude=='default' else yarn
    rot.inv_freq=src.inv_freq.detach().clone()
    rot.original_inv_freq=src.original_inv_freq.detach().clone()
    rot.attention_scaling=scale.attention_scaling
    if not torch.equal(rot.inv_freq,src.inv_freq):raise RuntimeError('RoPE frequencies did not preserve source')
    if rot.attention_scaling!=scale.attention_scaling:raise RuntimeError('RoPE scale did not preserve source')
    return rot

def score(model,tok,head,rows):
    tok.padding_side='right'
    scored={}
    for start in range(0,len(rows),BATCH):
        if ram_gib()<22:raise RuntimeError('Host available memory below 22GiB')
        group=rows[start:start+BATCH]
        tokens=tok([x['prompt'] for x in group],return_tensors='pt',padding=True,truncation=False)
        tokens.pop('token_type_ids',None)
        if tokens['input_ids'].shape[1]>135:raise RuntimeError('Unexpected token length')
        last=tokens['attention_mask'].sum(dim=1)-1
        with torch.inference_mode():
            hidden=model.model(**tokens,use_cache=False).last_hidden_state
            margin=(hidden[torch.arange(len(group)),last,:].float()@head.T)
            values=(margin[:,0]-margin[:,1]).cpu().numpy()
        for row,m in zip(group,values):
            key=(row['group'],row['state'],row['role'],row['condition'],row['order'])
            if key in scored:raise RuntimeError('Duplicate source event')
            signed=float(m) if row['correct']=='A' else -float(m)
            scored[key]={'signed':signed,'raw':float(m),'correct':int(signed>0),
                          'neutral':row['condition']=='neutral','role':row['role']}
        del tokens,hidden,margin
    if len(scored)!=256:raise RuntimeError('Incomplete 256 prompts')
    return scored

def episodes(scores):
    result=[]
    for group in range(16):
        for state in (0,1):
            for role in ('actor','second_party'):
                factual=[scores[group,state,role,'factual',order] for order in (0,1)]
                neutral=[scores[group,state,role,'neutral',order] for order in (0,1)]
                f=float(np.mean([x['signed'] for x in factual]))
                n=float(np.mean([x['signed'] for x in neutral]))
                result.append({'group':group,'role':role,
                    'raw_correct':sum(x['correct'] for x in factual),
                    'both_correct':int(all(x['correct'] for x in factual)),
                    'debiased_correct':int(f>0),'positive_fact_signal':int(f>n),
                    'fact_margin':f-n,
                    'factual_A':sum(x['raw']>0 for x in factual),
                    'neutral_A':sum(x['raw']>0 for x in neutral)})
    if len(result)!=64:raise RuntimeError('Expected 64 role-state episodes')
    return result

def aggregate(eps):
    def result(sub):
        return {k:sum(r[k] for r in sub) for k in METRICS} | {
            'n_episodes':len(sub),
            'mean_fact_margin':float(np.mean([r['fact_margin'] for r in sub])),
            'neutral_A_count':sum(r['neutral_A'] for r in sub),
            'factual_A_count':sum(r['factual_A'] for r in sub)}
    return {'overall':result(eps),
            'roles':{r:result([x for x in eps if x['role']==r])
                     for r in ('actor','second_party')}}

def boot(a,b,metric,seed=20261027):
    rng=np.random.default_rng(seed)
    by_group=[]
    for g in range(16):
        aa=[e[metric] for e in a if e['group']==g]
        bb=[e[metric] for e in b if e['group']==g]
        if len(aa)!=len(bb) or len(aa)!=4:raise RuntimeError('Unpaired source group')
        by_group.append(float(np.mean(bb)-np.mean(aa)))
    values=np.array(by_group)
    draws=rng.integers(0,16,(2000,16))
    estimate=values[draws].mean(axis=1)
    lo,hi=np.quantile(estimate,[0.025,0.975])
    return values.mean(),lo,hi

def render(source,scale_values,frequency_change,summaries,private_eps,hashes):
    r=['# EXP-027 — What drives K2 YaRN: rotary frequency or attention scaling?',
       '', '**October 14, 2026 | measured frozen BF16, independent authored source fixture.**',
       'Only the positional module was changed in memory; all K2 learned weights remained unchanged.',
       'This is synthetic A/B task behavior, not human-reviewed semantic competence.',
       '', '## Reproducibility and exact identities','',
       f"Fixture SHA256: `{source['sha256']}`. Novel name pairs 16, families 8, 256 unique prompts.",
       f'Preregistered protocol SHA256: `{PROTOCOL_SHA}`.',
       f'EXP018 authority SHA256: `{EXP018_SHA}`.',
       f'EXP025 full-model learned-weight report SHA256: `{PRIOR_WEIGHT_SHA}`.',
       f"K2 mid2 official revision: `{hashes['revision']}`.",
       f"Default rotational attention scaling: **{scale_values[0]:.9f}**; YaRN scaling: **{scale_values[1]:.9f}**.",
       f"Inverse frequencies differ in **{frequency_change}** positions; both rotary configs are pinned official code.",
       'All answers are conditional answer-token logit comparisons, NOT free-form generation.',
       '', '## Frozen learned weights, 2×2 RoPE-component factorial','',
       '| Inverse frequency | Cos/sin attention scaling | Raw factual correct /128 | Both orders correct /64 | Debiased correct entities /64 | Positive fact-reference signal /64 | Neutral A chosen /128 |',
       '|---|---|---:|---:|---:|---:|---:|']
    for key in CORNERS:
        entry=summaries[key]['overall']
        r.append(f"| {key[0]} | {key[1]} | {entry['raw_correct']} | {entry['both_correct']} | {entry['debiased_correct']} | {entry['positive_fact_signal']} | {entry['neutral_A_count']} |")
    r+=['','## Actor / recipient breakdown','',
        '| Frequency | Scaling | Role | Factual correct /64 | Both orders /32 | Debiased /32 | Positive fact signal /32 |',
        '|---|---|---|---:|---:|---:|---:|']
    for key in CORNERS:
        for role in ('actor','second_party'):
            entry=summaries[key]['roles'][role]
            r.append(f"| {key[0]} | {key[1]} | {role} | {entry['raw_correct']} | {entry['both_correct']} | {entry['debiased_correct']} | {entry['positive_fact_signal']} |")
    r+=['','## Controlled main contrasts and interaction','',
        'Fixed-weight change in output under different position-embedding components.',
        'Differences below are in **counts**; these are not independent data points.',
        '| Metric | Frequency-only delta | Scaling-only delta | Full YaRN delta | Interaction (full−frequency−scaling+default) |',
        '|---|---:|---:|---:|---:|']
    baseline=summaries['default','default']['overall']
    freq=summaries['yarn','default']['overall']
    scale=summaries['default','yarn']['overall']
    full=summaries['yarn','yarn']['overall']
    for m in METRICS:
        r.append(f'| {m} | {freq[m]-baseline[m]:+d} | {scale[m]-baseline[m]:+d} | {full[m]-baseline[m]:+d} | {full[m]-freq[m]-scale[m]+baseline[m]:+d} |')
    r+=['','### Paired 16-source-group descriptive resampling','',
        'Two groups share each of eight relation templates. The 2000 fixed-seed',
        'cluster-resampled 95% ranges are descriptive, not population intervals.',
        '| Contrast | Metric | Delta fraction per factual role-state | 95% descriptive group range |',
        '|---|---|---:|---:|']
    for label,key in [('frequency-only',('yarn','default')),
                      ('scaling-only',('default','yarn')),
                      ('full-YaRN',('yarn','yarn'))]:
        for name in ('both_correct','debiased_correct','positive_fact_signal'):
            v,lo,hi=boot(private_eps['default','default'],private_eps[key],name)
            r.append(f'| {label} | {name} | {v:+.4f} | [{lo:+.4f}, {hi:+.4f}] |')
    r+=['','## Interpretation and remaining gates','',
        'Rotary embedding frequencies change the spatial phase schedule;',
        'cosine and sine scaling multiplies the positional embedding components.',
        'Their product also affects attention-logit scaling and the nonlinear',
        'softmax. Both factors are applied identically to the same frozen K2 weights.',
        'A metric delta should not be assumed to be a pure semantic capability gain:',
        'answer-choice priors can dominate raw A/B correctness. Report each',
        'bias-corrected and neutral-reference metric independently.',
        'Any interaction indicates nonadditive behavior in this prompt distribution.',
        'This experiment does NOT identify a training teacher or prove general',
        'reasoning, unrelated-model parentage, or any production performance.',
        'Independent human labels, full-name generation, variable-length position',
        'controls and unrelated architecture baselines remain separate HOLD gates.',
        'Original model weights, checkpoints, inference services, network routes,',
        'graph data and NAS were not changed.',
        '', '## Exact replay','',
        'From the isolated scottjoyner/lms research branch, with pinned existing BF16 K2 assets:',
        '', '```bash',
        'python research/k2_yarn_components.py --evidence-dir /home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026 --out /home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026/EXPERIMENT-027-YARN-COMPONENTS.md',
        '```','']
    return '\n'.join(r)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise SystemExit('STOP: output exists, no overwrite')
    if ram_gib()<38:raise RuntimeError('Admission HOLD: available RAM below 38GiB')
    if sha(args.evidence_dir/'EXPERIMENT-027-PROTOCOL.md')!=PROTOCOL_SHA:raise RuntimeError('Protocol drift')
    if sha(args.evidence_dir/'results_exp018_intermediate_k2_20261014.json')!=EXP018_SHA:raise RuntimeError('Historical source drift')
    if sha(args.evidence_dir/'EXPERIMENT-025-FULL-WEIGHT-DRIFT.md')!=PRIOR_WEIGHT_SHA:raise RuntimeError('Model custody drift')
    sys.path.insert(0,str(args.evidence_dir))
    import run_exp018_k2_intermediate_gates as e18
    import run_exp016_k2_pretrain_release as e16
    import run_exp015_role_qa as qa
    import exp015_fixture
    fixture.check_disjoint()
    rows=fixture.make()
    source=fixture.receipt(rows)
    if source['sha256']!=FIXTURE_SHA:raise RuntimeError('New fixture drift')
    reference=json.loads((args.evidence_dir/'results_exp018_intermediate_k2_20261014.json').read_text())
    stage='mid2_47684'
    rec=reference['intermediate_config_and_index_receipts'][stage]
    path=e18.stage_path(stage)
    if e18.file_hash(path/'config.json')!=rec['sha256_config'] or e18.file_hash(path/'model.safetensors.index.json')!=rec['sha256_index']:
        raise RuntimeError('Pinned model metadata mismatch')
    base_tok=AutoTokenizer.from_pretrained(str(e16.BASE),local_files_only=True,trust_remote_code=False)
    model,tok,parity=e18.load_stage(stage,base_tok,exp015_fixture.make())
    if parity!=256 or sum(tok.encode(r['prompt'],add_special_tokens=False)==base_tok.encode(r['prompt'],add_special_tokens=False) for r in rows)!=256:
        raise RuntimeError('Tokenizer full prompt parity')
    head=model.lm_head.weight[qa.token_ids(tok),:].float().clone()
    core=model.model
    before=weight_receipts(core)
    orig=core.rotary_emb
    cfg=AutoConfig.from_pretrained(str(e18.stage_path('rl_merged')),local_files_only=True,trust_remote_code=True)
    if orig.rope_type!='default' or cfg.rope_parameters['rope_type']!='yarn':raise RuntimeError('Native RoPE identities mismatch')
    yarn=type(orig)(config=cfg,device=torch.device('cpu'))
    if torch.equal(orig.inv_freq,yarn.inv_freq):raise RuntimeError('Identical default/YaRN frequencies')
    if orig.attention_scaling==yarn.attention_scaling:raise RuntimeError('Identical default/YaRN amplitudes')
    frequency_change=int(torch.count_nonzero(orig.inv_freq!=yarn.inv_freq))
    summaries={};source_episodes={}
    try:
        for key in CORNERS:
            rotary=setup_rope(orig,yarn,*key)
            core.rotary_emb=rotary
            raw=score(model,tok,head,rows)
            episodes_=episodes(raw)
            summaries[key]=aggregate(episodes_)
            source_episodes[key]=episodes_
            print('EXP027_CONDITION',key,summaries[key]['overall'],flush=True)
            del rotary,raw,episodes_
            gc.collect()
    finally:
        core.rotary_emb=orig
    if weight_receipts(core)!=before:raise RuntimeError('Learned K2 weights modified')
    if len(summaries)!=4:raise RuntimeError('Incomplete factorial')
    content=render(source,(float(orig.attention_scaling),float(yarn.attention_scaling)),
                   frequency_change,summaries,source_episodes,
                   {'revision':rec['revision']})
    if not args.out.parent.is_dir():raise RuntimeError('Destination parent absent')
    temp=args.out.with_name(args.out.name+'.tmp')
    if temp.exists():raise RuntimeError('Temporary output collision')
    temp.write_text(content,encoding='utf-8')
    temp.replace(args.out)
    print('EXP027_COMPLETE',len(content.splitlines()),sha(args.out),flush=True)

if __name__=='__main__':main()
