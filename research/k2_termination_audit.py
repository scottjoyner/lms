#!/usr/bin/env python3
"""EXP031: read-only, eight-prompt native K2 special-terminator audit."""
import argparse
import hashlib
import os
import sys
import time
from pathlib import Path
for k,v in {'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
            'CUDA_VISIBLE_DEVICES':'-1','OPENBLAS_NUM_THREADS':'1',
            'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}.items():os.environ[k]=v
import torch

PROTOCOL='a66ab26753687ee5ef2803ee2d38c259bfb1f93eddb4b6a3940ea28d926367aa'
PROMPT_SHA='6513d65b456a0f52d230e386078d732a4b2800663c4d9b11c76b3cd569651c2a'
GOLD_SHA='aad6e578544e2a3f100ab4b5c93d7ee6158fbfbe14bd426b8bffd9f9d1cde23b'
START=time.monotonic()
END_IDS={'ifm_message_end':64019,'eot_id':64024,
         'end_of_thought':64026,'end_of_solution':64028,'assistant_end':64040}
SOURCE_GROUPS=(0,1,2,3)
MAX_NEW=32

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def memory():
    return next(int(s.split()[1]) for s in Path('/proc/meminfo').read_text().splitlines()
                if s.startswith('MemAvailable:'))/1024**2

def preflight(evidence,prior,tok):
    if sha(evidence/'EXPERIMENT-031-PROTOCOL.md')!=PROTOCOL:
        raise RuntimeError('Terminator audit protocol drift')
    cases,receipt,eligible=prior.source_rows()
    if receipt!=GOLD_SHA or eligible!=90:raise RuntimeError('SQuAD2 source changed')
    rows,original_sha,filler=prior.generate_rows(tok,cases,'mopd_249')
    if original_sha!=PROMPT_SHA or filler!=768:
        raise RuntimeError('Original EXP029 exact generation input changed')
    selected=[r for r in rows if r['position']=='short' and r['article_index'] in SOURCE_GROUPS]
    if len(selected)!=8 or sum(r['impossible'] for r in selected)!=4:
        raise RuntimeError('Expected exactly four article pairs')
    assert all(r['position']=='short' for r in selected)
    actual={key:tok.convert_tokens_to_ids(text) for key,text in (
      ('ifm_message_end','<|ifm|im_end|>'),('eot_id','<|eot_id|>'),
      ('end_of_thought','<|end_of_thought|>'),
      ('end_of_solution','<|end_of_solution|>'),
      ('assistant_end','<|assistant_end|>'))}
    if actual!=END_IDS:raise RuntimeError('Official alternate stop-token mapping mismatch')
    if tok.eos_token_id!=1 or tok.pad_token_id!=64255:
        raise RuntimeError('K2 published EOS or PAD mismatch')
    return selected

def score_tokens(ids,eos_id,all_special):
    eos=[idx for idx,x in enumerate(ids) if x==eos_id]
    stop=eos[0] if eos else len(ids)
    prefix=ids[:stop]
    known={name:[i for i,v in enumerate(prefix) if v==token]
           for name,token in END_IDS.items()}
    other=[(i,v) for i,v in enumerate(prefix)
           if v in all_special and v not in END_IDS.values()]
    return {'eos':bool(eos),'eos_pos':eos[0] if eos else None,
            'cap':not eos and stop==MAX_NEW,'n_before_eos':stop,
            'marks':known,'other_special_count':len(other),
            'other_special_unique':len(set(v for _,v in other))}

def render(samples,revision,config_sha,tokenizer_sha):
    counts={name:sum(bool(r['marks'][name]) for r in samples) for name in END_IDS}
    totals={name:sum(len(r['marks'][name]) for r in samples) for name in END_IDS}
    lines=['# EXP-031 — Native K2 generation EOS and alternate termination witness',
        '', '**2026-10-14 | actual frozen 8-prompt read-only CPU audit.**',
        'Sampled first four previously frozen SQuAD2 dev source articles, one',
        'human-answerable and one human-unanswerable each, short-context only.',
        'Eight examples do NOT establish a population stop-token frequency.',
        '', '## Model and source custody','',
        f'Pre-inference protocol SHA256 `{PROTOCOL}`.',
        f'EXP029 exact selected human-gold source SHA256 `{GOLD_SHA}`.',
        f'EXP029 original 48-prompt tokenizer receipt `{PROMPT_SHA}`.',
        f'Frozen official K2 mOPD revision `{revision}`.',
        f'Model config SHA256 `{config_sha}`; tokenizer SHA256 `{tokenizer_sha}`.',
        'Tokenizer, model config, and generation config native EOS ID **1**',
        '(`<|endoftext|>`), padded with token ID **64255**.',
        f'Greedy direct generation, {MAX_NEW} max new tokens, no manual stop override.',
        '', '## Exact observed termination/candidate marker inventory','',
        '| Marker | Native token ID | Sequences containing before EOS /8 | Total emitted before EOS | Observed first-position min–max |',
        '|---|---:|---:|---:|---:|']
    for name,token in END_IDS.items():
        positions=[pos for r in samples for pos in r['marks'][name]]
        locus=(f'{min(positions)}–{max(positions)}' if positions else 'not observed')
        lines.append(f'| {name} | {token} | {counts[name]} | {totals[name]} | {locus} |')
    eos=sum(r['eos'] for r in samples)
    caps=sum(r['cap'] for r in samples)
    other=sum(r['other_special_count'] for r in samples)
    lines+=['',f'Native EOS observed: **{eos}/8**.',
            f'All {MAX_NEW} new tokens exhausted without native EOS: **{caps}/8**.',
            f'Other noncandidate special tokens seen before EOS: **{other}** occurrences.',
            f'Any five audited alternate terminators: **{sum(any(r["marks"].values()) for r in samples)}/8**.',
            '', '## Interpretation and explicit non-release boundary','',
            'This only checks whether the given native K2 delimiter IDs appear',
            'during original-format generation, not whether they should be',
            'considered correct completion markers. An alternate marker might',
            'also occur inside a legitimate thought/code/tool segment and must',
            'not be promoted to production stop semantics without independent',
            'verification and negative tests. Missing markers cannot establish',
            'that none occur for all prompts. Eight public dev examples are tiny.',
            'No human prompt or raw token stream is exported. No model weights,',
            'training, NAS, graph, fleet model registry or routing were changed.',
            '', '## Reproduce','',
            'Use frozen SQuAD2 and local BF16 model with:',
            '`python research/k2_termination_audit.py --evidence-dir /home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026 --out /home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026/EXPERIMENT-031-TERMINATORS.md`','']
    return '\n'.join(lines)

def main():
    c=argparse.ArgumentParser(description=__doc__)
    c.add_argument('--evidence-dir',type=Path,required=True)
    c.add_argument('--out',type=Path,required=True)
    a=c.parse_args()
    if a.out.exists():raise SystemExit('STOP output collision')
    sys.path.insert(0,str(a.evidence_dir))
    import k2_human_grounded_qa as prior
    import run_exp021_free_response as backend
    from transformers import AutoTokenizer
    import run_exp018_k2_intermediate_gates as exp18
    tok=AutoTokenizer.from_pretrained(str(exp18.stage_path('mopd_249')),
                                       local_files_only=True,trust_remote_code=False)
    rows=preflight(a.evidence_dir,prior,tok)
    if memory()<36:raise RuntimeError('Resource HOLD below 36 GiB initial RAM')
    model,tok,receipt=backend.load_model('mopd_249')
    try:
        if receipt['revision']!='7abae15437c2e7f1b2cd986be562344f204a17f3':
            raise RuntimeError('Wrong K2 native model revision')
        if (model.config.eos_token_id!=tok.eos_token_id or
            model.generation_config.eos_token_id!=tok.eos_token_id):
            raise RuntimeError('Model/generation config EOS drift')
        all_special=set(tok.all_special_ids)
        tok.padding_side='left'; scored=[]
        for i in range(0,len(rows),2):
            if memory()<22:raise RuntimeError('Memory safety stop below 22 GiB')
            batch=rows[i:i+2]
            input=tok([r['prompt'] for r in batch],padding=True,
                      return_tensors='pt',truncation=False)
            input.pop('token_type_ids',None)
            with torch.inference_mode():
                generated=model.generate(**input,do_sample=False,num_beams=1,
                    max_new_tokens=MAX_NEW,use_cache=True,
                    eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id)
            for ids in generated[:,input['input_ids'].shape[1]:]:
                scored.append(score_tokens(ids.tolist(),tok.eos_token_id,all_special))
            print('EXP031_GROUP',i+len(batch),'/',len(rows),
                  'elapsed_sec',round(time.monotonic()-START,2),flush=True)
            del input,generated
        if len(scored)!=8:raise RuntimeError('Incomplete terminator audit')
        output=render(scored,receipt['revision'],receipt['model_config_sha256'],
                      receipt['tokenizer_sha256'])
        scratch=a.out.with_name(a.out.name+'.tmp')
        if scratch.exists():raise RuntimeError('Scratch output collision')
        scratch.write_text(output,encoding='utf8');scratch.replace(a.out)
        print('EXP031_COMPLETE',sha(a.out),'eos',sum(r['eos'] for r in scored),
              'caps',sum(r['cap'] for r in scored),'time_s',round(time.monotonic()-START,2))
    finally:
        del model
if __name__=='__main__':main()
