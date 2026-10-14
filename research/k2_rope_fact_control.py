#!/usr/bin/env python3
"""EXP026: K2 2x2 RoPE-weight intervention on EXP019 fact + neutral controls.
Frozen CPU, pinned sources; outputs only human-readable Markdown.
"""
import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path

for k,v in {'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
            'CUDA_VISIBLE_DEVICES':'-1','OPENBLAS_NUM_THREADS':'1',
            'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}.items():os.environ[k]=v
import torch
from transformers import AutoTokenizer,AutoConfig
from k2_rope_factorial import weight_receipts

STAGES=('mid2_47684','rl_merged')
ROPES=('default','yarn')
PRIOR_SHA='12546abb9ed90e4e51630a42415a0cd801bb0642db8a375f706e960ed61fcaf6'
CORPUS_SHA='8945e221632a3ed30a9d22176a8d7222f4f9ee09aa3e0354a4937fd2f20c27c8'
FIELDS=('factual_raw_correct','factual_pair_both_correct',
        'factual_debiased_correct','fact_signal_positive')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def md_table(summary):
    lines=['| Frozen weights | RoPE | Raw correct /256 | Both orders /128 | Debiased /128 | Positive fact signal /128 | Neutral A rate |',
           '|---|---|---:|---:|---:|---:|---:|']
    for key in STAGES:
        for rope in ROPES:
            t=summary[key,rope]['overall']
            lines.append(f"| {key} | {rope} | {t['factual_raw_correct']} | {t['factual_pair_both_correct']} | {t['factual_debiased_correct']} | {t['fact_signal_positive']} | {100*t['raw_neutral_A_rate']:.2f}% |")
    return '\n'.join(lines)

def render(summary,differences):
    s=['# EXP-026 — K2 RoPE ablation on new factual and neutral relationship templates',
       '', '**2026-10-14 | actual frozen factorial, all 512 EXP-019 prompts per condition.**',
       'This second synthetic fixture tests transport of EXP-024 conclusions,',
       'not generalization to independent human-reviewed examples.',
       '', '## Source custody and frozen design','',
       f'Prior EXP019 SHA256: `{PRIOR_SHA}`.',
       f'Pinned authored stimulus SHA256: `{CORPUS_SHA}`.',
       'Two official K2 checkpoint weights; same verified tokenizer and 255 weights.',
       'Only in-memory rotary embedding module changes between conditions.',
       'Four conditions × 512 prompts, 256 factual and 256 neutral each.',
       'Underlying EXP-019 source has 32 related surname-pair groups,',
       'eight relation templates, reversals, two question roles and both A/B orders.',
       '', '## Five targeted answer and fact-dependence diagnostics','',md_table(summary),
       '', '## Actor versus recipient paired consistency','',
       '| Weight stage | RoPE | Actor both-order correct /64 | Recipient both-order correct /64 |',
       '|---|---|---:|---:|']
    for key in STAGES:
        for rope in ROPES:
            r=summary[key,rope]['role']
            s.append(f"| {key} | {rope} | {r['actor']['factual_pair_both_correct']} | {r['second_party']['factual_pair_both_correct']} |")
    s+=['','## Factorial contrasts and paired group-resampled ranges','',
        'Each contrast uses the same 32 surname groups, not 128 independent',
        'episodes. Fixed seed, 2000 resamples; 95% ranges are descriptive',
        'only and not population confidence intervals.', '']
    names={'mid2_rope':(('mid2_47684','default'),('mid2_47684','yarn')),
           'rl_rope':(('rl_merged','default'),('rl_merged','yarn')),
           'default_weights':(('mid2_47684','default'),('rl_merged','default')),
           'yarn_weights':(('mid2_47684','yarn'),('rl_merged','yarn')),
           'native_transition':(('mid2_47684','default'),('rl_merged','yarn'))}
    metrics={'factual_raw_correct':'Raw factual accuracy',
             'factual_pair_both_correct':'Both option orders correct',
             'factual_debiased_correct':'Debiased entity accuracy',
             'fact_signal_positive':'Positive fact-minus-neutral signal',
             'fact_signal_margin':'Mean signed fact contribution'}
    s+=['| Controlled difference | Metric | Delta | Descriptive 95% range |',
        '|---|---|---:|---:|']
    for tag in names:
        for code,label in metrics.items():
            r=differences[tag][code]
            lo,hi=r['group_bootstrap_95_descriptive']
            s.append(f"| {tag} | {label} | {r['observed_delta']:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    s+=['','## Factorial interaction','',
        'Difference in the RoPE effect between RL and mid2 weights:',
        '| Metric | Interaction (RL RoPE effect − mid2 RoPE effect) |',
        '|---|---:|']
    for key in FIELDS:
        a=summary['mid2_47684','default']['overall'][key]
        b=summary['mid2_47684','yarn']['overall'][key]
        c=summary['rl_merged','default']['overall'][key]
        d=summary['rl_merged','yarn']['overall'][key]
        s.append(f'| {metrics[key]} | {(d-c)-(b-a):+d} count(s) |')
    s+=['','## Interpretation limits','',
        'RoPE effects at fixed learned weights are controlled forward-pass',
        'interventions. Checkpoint weight comparisons remain observational.',
        'Neutral statements have no correct answer; any signed neutral',
        'reference margin is a diagnostic, not neutral-question accuracy.',
        'A/B priors and signed contrast score inversion can distort conclusions.',
        'The EXP019 data was already inspected while choosing this hypothesis:',
        'this is a new controlled *format* replication, not independent',
        'pre-registered population evidence. Independent human labels are absent.',
        'No semantic teacher-lineage claim, model promotion, production',
        'router, graph, NAS, or live service action is authorized.',
        '', '## Reproduce','',
        '`python research/k2_rope_fact_control.py --evidence-dir /home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026 --out /home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026/EXPERIMENT-026-FACT-ROPE.md`', '']
    return '\n'.join(s)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise SystemExit('STOP output exists')
    sys.path.insert(0,str(a.evidence_dir))
    import run_exp019_fact_bias as e19
    import run_exp015_role_qa as qa
    import run_exp018_k2_intermediate_gates as e18
    import run_exp016_k2_pretrain_release as e16
    if e19.mem_gib()<38:raise RuntimeError('Host resource admission HOLD <38GiB')
    src=a.evidence_dir/'results_exp019_fact_bias_20261014.json'
    if sha(src)!=PRIOR_SHA:raise RuntimeError('Original EXP019 SHA custody mismatch')
    ref=json.loads(src.read_text())
    rows=e19.fixture.make()
    if e19.fixture.receipt(rows)['source_sha256']!=CORPUS_SHA:raise RuntimeError('EXP019 fixture changed')
    oldrows=e19.oldfixture.make()
    base=AutoTokenizer.from_pretrained(str(e16.BASE),local_files_only=True,trust_remote_code=False)
    config={rope:AutoConfig.from_pretrained(str(e18.stage_path(
             'mid2_47684' if rope=='default' else 'rl_merged')),
             local_files_only=True,trust_remote_code=True) for rope in ROPES}
    assert [config[z].rope_parameters['rope_type'] for z in ROPES]==['default','yarn']
    score={};episodes={}
    for stage in STAGES:
        if ref['model_commits'][stage]!=next(x[1] for x in e18.STAGES if x[0]==stage):
            raise RuntimeError('Checkpoint commit disagreement')
        model,tok,parity=e18.load_stage(stage,base,oldrows)
        if parity!=256:raise RuntimeError('Stage tokenizer parity mismatch')
        core=model.model
        oldrot=core.rotary_emb
        before=weight_receipts(core)
        head=model.lm_head.weight[qa.token_ids(tok),:].float().clone()
        try:
            for rope in ROPES:
                if e19.mem_gib()<22:raise RuntimeError('Host in-step memory gate failed')
                newrot=type(oldrot)(config=config[rope],device=torch.device('cpu'))
                core.rotary_emb=newrot
                rec=e19.classify('k2',model,tok,head,rows)
                eps=e19.aggregate(rec)
                sums=e19.summary(eps)
                native=(rope=='default' if stage=='mid2_47684' else rope=='yarn')
                if native:
                    expected=ref['stages'][stage]['overall']
                    for field in FIELDS:
                        if sums['overall'][field]!=expected[field]:
                            raise RuntimeError(f'Native EXP019 count mismatch {stage} {field}')
                score[stage,rope]=sums
                episodes[stage,rope]=eps
                print('EXP026_STAGE',stage,rope,{k:sums['overall'][k] for k in FIELDS},flush=True)
                del rec,eps,sums,newrot
                gc.collect()
        finally:
            core.rotary_emb=oldrot
        if weight_receipts(core)!=before:raise RuntimeError('Frozen model weights changed')
        del model,tok,head,core,oldrot
        gc.collect()
    pairs={
      'mid2_rope':(('mid2_47684','default'),('mid2_47684','yarn')),
      'rl_rope':(('rl_merged','default'),('rl_merged','yarn')),
      'default_weights':(('mid2_47684','default'),('rl_merged','default')),
      'yarn_weights':(('mid2_47684','yarn'),('rl_merged','yarn')),
      'native_transition':(('mid2_47684','default'),('rl_merged','yarn')),
    }
    deltas={key:e19.bootstrap_pair_diff(episodes[left],episodes[right])
            for key,(left,right) in pairs.items()}
    text=render(score,deltas)
    if not a.out.parent.is_dir():raise RuntimeError('Destination parent absent')
    tmp=a.out.with_name(a.out.name+'.tmp')
    if tmp.exists():raise RuntimeError('Output temporary collision')
    tmp.write_text(text,encoding='utf-8')
    tmp.replace(a.out)
    print('EXP026_PASS',len(text.splitlines()),hashlib.sha256(text.encode()).hexdigest(),flush=True)

if __name__=='__main__':main()
