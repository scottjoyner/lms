#!/usr/bin/env python3
"""Native K2 rotary phase diagnostic, analytic only; no model inference."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import torch
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

STAGES=('mid2_47684','rl_merged')
DISTANCES=(1,8,32,128,384,1024,4096,8192,32768,131072)
REF_SHA='3818a421a9b9cbe6f52b0a27d8c3da69be0ee1bd4d50071c190d50ba95ee893a'
torch.set_num_threads(2)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def phase_diagnostics(a,b,distances=DISTANCES):
    if a.shape!=b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('Invalid inverse-frequency tensors')
    if a.numel()!=32:raise ValueError('Unexpected K2 rotary dimensions')
    rows=[]
    for gap in distances:
        phase=gap*(b.double()-a.double())
        wrapped=torch.atan2(phase.sin(),phase.cos())
        abs_gap=wrapped.abs()
        # RMS chord distance between same unit-circle basis vectors after
        # relative d rotations. It is NOT model activation CKA or task accuracy.
        rms=math.sqrt(float(torch.mean(2-2*phase.cos())))
        rows.append((gap,float(abs_gap.mean()),float(abs_gap.max()),rms,
                    int(torch.count_nonzero(abs_gap>math.pi/2))))
    return rows

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-dir',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    x=parser.parse_args()
    if x.out.exists():raise SystemExit('STOP output exists')
    prior=x.evidence_dir/'results_exp018_intermediate_k2_20261014.json'
    if sha(prior)!=REF_SHA:raise RuntimeError('Source receipt mismatch')
    rec=json.loads(prior.read_text())['intermediate_config_and_index_receipts']
    modes={}
    for stage in STAGES:
        p=Path(f'/home/scott/git/.latent-k2-{stage}-research')
        if sha(p/'config.json')!=rec[stage]['sha256_config']:
            raise RuntimeError('Native K2 config altered: '+stage)
        cfg=AutoConfig.from_pretrained(str(p),local_files_only=True,trust_remote_code=True)
        cls=get_class_from_dynamic_module('modeling_k2_horizon.K2HorizonRotaryEmbedding',str(p),local_files_only=True)
        rotary=cls(cfg,device=torch.device('cpu'))
        modes[stage]=(cfg,rotary)
    a=modes[STAGES[0]][1];b=modes[STAGES[1]][1]
    if a.rope_type!='default' or b.rope_type!='yarn':raise RuntimeError('Wrong compared native modes')
    if abs(float(a.attention_scaling)-1)>1e-12 or abs(float(b.attention_scaling)-1.2772588722239782)>1e-10:
        raise RuntimeError('Rotary amplitude custody mismatch')
    delta=int(torch.count_nonzero(a.inv_freq!=b.inv_freq))
    if delta!=26:raise RuntimeError('Unexpected changed native rotary frequencies')
    rows=phase_diagnostics(a.inv_freq,b.inv_freq)
    lines=['# EXP-028A — Native K2 rotary phase differences versus relative distance',
        '', '**October 14, 2026 | analytic native-module diagnostic; NO neural inference.**',
        'This is a support/control for EXP-028, not a completed long-context model test.',
        '', '## Exact source identity','',
        f'EXP018 model custody SHA256 `{REF_SHA}`.',
        f"K2 mid2 default native config SHA256 `{rec[STAGES[0]]['sha256_config']}`.",
        f"K2 RL native YaRN config SHA256 `{rec[STAGES[1]]['sha256_config']}`.",
        f'Native rotary module has 32 inverse-frequency channels, **{delta}** differ.',
        f'Default amplitude {float(a.attention_scaling):.9f}; YaRN amplitude {float(b.attention_scaling):.9f}.',
        'The phase-only comparison normalizes away amplitude by comparing',
        'unit-circle rotations after a given relative token distance.',
        '', '## Native frequency schedule disagreement by relative token gap','',
        '| Relative token distance | Mean absolute wrapped phase difference (rad) | Maximum wrapped difference (rad) | Unit-circle RMS chord distance | Channels differing >π/2 |',
        '|---:|---:|---:|---:|---:|']
    for g,mean,mx,rms,num in rows:
        lines.append(f'| {g} | {mean:.6f} | {mx:.6f} | {rms:.6f} | {num}/32 |')
    lines+=['', '## What can and cannot be concluded','',
        'YaRN frequency schedules need not have the same phases at token',
        'separations even where their A/B task outputs happen to agree.',
        'The table is computed from native rotary inverse frequencies only:',
        'it does not include Q/K projections, attention gates, nonlinear',
        'softmax or model semantic behavior. A large phase difference is not',
        'proof of improved or degraded context retrieval.',
        'EXP028 tests actual frozen-model logits at length-matched prompt',
        'placements only when host memory and bounded-run gates pass.',
        'This diagnostic makes no claim about K2 teacher lineage, long-context',
        'real-world competence, checkpoint promotion or release readiness.',
        '', '## Reproduction', '', '```bash',
        'python research/k2_rotary_phase.py --evidence-dir /home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026 --out /home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026/EXPERIMENT-028A-ROTARY-PHASE.md',
        '```','']
    if not x.out.parent.is_dir():raise RuntimeError('Missing output directory')
    content='\n'.join(lines)
    tmp=x.out.with_name(x.out.name+'.tmp')
    if tmp.exists():raise RuntimeError('Existing staging path')
    tmp.write_text(content,encoding='utf8');tmp.replace(x.out)
    print('EXP028A_PHASE_PASS',sha(x.out),'rows',len(rows))

if __name__=='__main__':main()
