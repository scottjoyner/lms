#!/usr/bin/env python3
"""EXP032: real frozen-K2 32-trace, source-only posthoc readout controls.

Model output and human QA never persisted; aggregate Markdown is authoritative.
"""
import argparse
import hashlib
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

for key,value in {'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
                  'CUDA_VISIBLE_DEVICES':'-1','OPENBLAS_NUM_THREADS':'1',
                  'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}.items():os.environ[key]=value
import torch

PROTOCOL_SHA='320b31b3bf07600e22e6246e411779c89563a6be18291bdd21ee2ec4c940a665'
SOURCE_SELECTION='aad6e578544e2a3f100ab4b5c93d7ee6158fbfbe14bd426b8bffd9f9d1cde23b'
EXP029_PROMPTS='6513d65b456a0f52d230e386078d732a4b2800663c4d9b11c76b3cd569651c2a'
EXP029_RESULTS='0ab5ac2d744362969b97618b32ce9ad536ae32bda820a4bfac79410cc7c17c71'
EXP031_RESULTS='a0e7db3d681eb81cc3880715aab9f9542948a4008b1e61516ad7e8c1499beb50'
MODEL_REV='7abae15437c2e7f1b2cd986be562344f204a17f3'
CASES=32
MAX_NEW=32
ALT_TOKEN=64019
STRATEGIES=('native','alt_cut','first_word','lexical_gate','question_exclusion_gate','always_unknown')
PLACES=('short','gap')
START=time.monotonic()
torch.set_num_threads(2)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def available_gib():
    return next(int(x.split()[1]) for x in Path('/proc/meminfo').read_text().splitlines()
                if x.startswith('MemAvailable:'))/1024**2

def preflight(directory,prior,tok):
    if sha(directory/'EXPERIMENT-032B-STOP-POSTHOC-PROTOCOL.md')!=PROTOCOL_SHA:
        raise RuntimeError('EXP032B protocol changed')
    if sha(directory/'EXPERIMENT-029-K2-HUMAN-QA.md')!=EXP029_RESULTS:
        raise RuntimeError('Original human-gold baseline changed')
    if sha(directory/'EXPERIMENT-031-TERMINATORS.md')!=EXP031_RESULTS:
        raise RuntimeError('Original delimiter witness changed')
    source,receipt,eligible=prior.source_rows()
    if receipt!=SOURCE_SELECTION or eligible!=90:
        raise RuntimeError('SQuAD2 human-gold selection changed')
    all_rows,prompt_receipt,filler=prior.generate_rows(tok,source,'mopd_249')
    if prompt_receipt!=EXP029_PROMPTS or filler!=768:
        raise RuntimeError('SQuAD2 original frozen input prompt mismatch')
    if tok.eos_token_id!=1 or tok.convert_tokens_to_ids('<|ifm|im_end|>')!=ALT_TOKEN:
        raise RuntimeError('Published token IDs changed')
    rows=[r for r in all_rows if r['position'] in PLACES]
    if len(rows)!=CASES or len({(r['qa_id'],r['position']) for r in rows})!=CASES:
        raise RuntimeError('Invalid 32-prompt cohort')
    for place in PLACES:
        group=[r for r in rows if r['position']==place]
        if len(group)!=16 or sum(r['impossible'] for r in group)!=8:
            raise RuntimeError('Invalid factual/unanswerable balance')
    return rows,receipt

def first_word(text):
    match=re.match(r'^\s*([A-Za-z][A-Za-z\-]*)',text)
    return match.group(1) if match else ''

def has_word(text,word):
    if not word:return False
    return bool(re.search(r'(?<![A-Za-z])'+re.escape(word)+r'(?![A-Za-z])',text,re.I))

def interpret(ids,tok,context,question):
    """Raw decoder is read-only; rules depend on source text only, NEVER gold."""
    eos=next((i for i,t in enumerate(ids) if t==tok.eos_token_id),len(ids))
    native_text=tok.decode(ids[:eos],skip_special_tokens=True)
    boundary=next((i for i,t in enumerate(ids[:eos]) if t==ALT_TOKEN),eos)
    alt_text=tok.decode(ids[:min(eos,boundary)],skip_special_tokens=True)
    candidate=first_word(alt_text)
    readouts={'native':native_text,'alt_cut':alt_text,
              'first_word':candidate if candidate else 'UNKNOWN',
              'always_unknown':'UNKNOWN'}
    eligible=(bool(candidate) and 2<=len(candidate)<=26 and has_word(context,candidate))
    readouts['lexical_gate']=candidate if eligible else 'UNKNOWN'
    readouts['question_exclusion_gate']=candidate if eligible and not has_word(question,candidate) else 'UNKNOWN'
    return readouts,{'eos':int(eos<len(ids)),
       'cap':int(eos==MAX_NEW and len(ids)>=MAX_NEW),
       'alt':int(boundary<eos),'alt_position':boundary if boundary<eos else None,
       'alt_changed':int(native_text!=alt_text)}

def summarize(rows,place,strategy):
    match=[r for r in rows if r['placement']==place and r['strategy']==strategy]
    a=[r for r in match if not r['unanswerable']]
    u=[r for r in match if r['unanswerable']]
    if len(a)!=8 or len(u)!=8:raise RuntimeError('Missing required source-scoring controls')
    return {'strict':sum(r['strict'] for r in a),
            'lenient':sum(r['lenient'] for r in a),
            'unknown':sum(r['strict'] for r in u),
            'negative_nonabstention':sum(r['nonabstention'] for r in u),
            'answer_unknown':sum(r['unexpected_unknown'] for r in a),
            'output_changed':sum(r['changed_from_native'] for r in match)}

def render(summary,markers,source_sha,model_receipt):
    lines=['# EXP-032B — K2 post-hoc stop-token and source-lexical readout',
           '', '**2026-10-14 | actual 32 frozen model generations; post-hoc policies only.**',
           'Original SQuAD2 human-annotated dev source from Rajpurkar et al. (2018), CC BY-SA 4.0;',
           'https://rajpurkar.github.io/SQuAD-explorer/ .',
           'This is a previously exposed public eight-article subset, NOT blinded human evaluation.',
           '', '## Exact custody and never-changing native generation','',
           f'EXP032B preregistered protocol SHA256 `{PROTOCOL_SHA}`.',
           f'Original human gold subset receipt `{source_sha}`.',
           f'Original EXP029 prompt/placement receipt `{EXP029_PROMPTS}`.',
           f'Original EXP029 generated-answer Markdown SHA256 `{EXP029_RESULTS}`.',
           f'Original EXP031 native delimiter witness SHA256 `{EXP031_RESULTS}`.',
           f"Frozen K2 official model revision `{model_receipt['revision']}`.",
           f"Model config SHA256 `{model_receipt['model_config_sha256']}`.",
           f"Tokenizer SHA256 `{model_receipt['tokenizer_sha256']}`.",
           'Greedy 32-token K2 plain-completion, untouched native EOS ID1;',
           'all policy changes are applied **after the same one-shot model trace**.',
           'No gold or is_unanswerable flag passed to any interpretation rule.',
           '', '## Human-gold strict behavior and false positives','',
           '| Source position | Post-hoc readout | Strict human gold /8 | Gold anywhere /8 | Exact UNKNOWN /8 negatives | Non-UNKNOWN /8 negatives | UNKNOWN on answerable /8 | Outputs different from native /16 |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for pos in PLACES:
        for arm in STRATEGIES:
            c=summary[pos][arm]
            lines.append(f"| {pos} | {arm} | {c['strict']} | {c['lenient']} | {c['unknown']} | {c['negative_nonabstention']} | {c['answer_unknown']} | {c['output_changed']} |")
    lines+=['','## What native K2 actually emitted','',
        '| Placement | Native EOS /16 | 32-token caps /16 | Alternate `<|ifm|im_end|>` before EOS /16 | Cases where alternate cut changes text /16 |',
        '|---|---:|---:|---:|---:|']
    for place in PLACES:
        group=markers[place]
        lines.append(f"| {place} | {sum(v['eos'] for v in group)} | {sum(v['cap'] for v in group)} | {sum(v['alt'] for v in group)} | {sum(v['alt_changed'] for v in group)} |")
    for place in PLACES:
        if summary[place]['native']['strict']!=0 or summary[place]['native']['unknown']!=0:
            raise RuntimeError('Frozen EXP029 strict/EOS baseline unexpectedly changed')
        if summary[place]['native']['lenient']!=({'short':7,'gap':6}[place]):
            raise RuntimeError('EXP029 previous source/frozen-run result mismatch')
        if sum(v['cap'] for v in markers[place])!=({'short':9,'gap':14}[place]):
            raise RuntimeError('EXP029 previous generation cap mismatch')
    lines+=['','## Safety-critical interpretation','',
           'An alternate marker in generated text is **not** proof that it means',
           'end of a completed answer. First-word and lexical occurrence are',
           'post-hoc formatter/substring controls, NOT independent evidence',
           'of relation entailment or a trusted UNKNOWN decision. A negative',
           'SQuAD item can contain every word of an incorrect proposed answer.',
           'An always-UNKNOWN strategy is expected to avoid unsupported answers',
           'only by sacrificing all answerable coverage; report it as a control.',
           'The dataset contains only eight reused human-annotated source articles,',
           'with correlated short and long-gap questions. No significance,',
           'teacher-ancestry, model-layer provenance or readiness claim follows.',
           'Do not change native EOS, register another stop marker in production,',
           'or promote a lexical gate based on this exploratory result.',
           '', '## Reproduction','',
           'Code and tests: `scottjoyner/lms` isolated research branch,',
           '`research/k2_posthoc_boundary.py` and `tests/test_k2_posthoc_boundary.py`.',
           'Same frozen original official checkpoint, dataset and host admissions.',
           'Generated token lists, raw answers and human passages stay in RAM.',
           'The experiment is safe to rerun using a **new** unused report path:',
           '', '```bash',
           'python research/k2_posthoc_boundary.py --evidence-dir /home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026 --out /home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026/EXPERIMENT-032B-STOP-POSTHOC.md',
           '```','']
    return '\n'.join(lines)

def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--evidence-dir',required=True,type=Path)
    cli.add_argument('--out',required=True,type=Path)
    cli.add_argument('--preflight-only',action='store_true')
    x=cli.parse_args()
    if x.out.exists():raise SystemExit('STOP existing report cannot be overwritten')
    if not x.out.parent.is_dir():raise RuntimeError('Missing report parent')
    sys.path.insert(0,str(x.evidence_dir))
    import k2_human_grounded_qa as prev
    import run_exp021_free_response as prior_model
    import run_exp018_k2_intermediate_gates as e18
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(e18.stage_path('mopd_249')),
                                            local_files_only=True,trust_remote_code=False)
    cases,receipt=preflight(x.evidence_dir,prev,tokenizer)
    print('EXP032B_SOURCE_MATCH',receipt,'n',len(cases),'mem_GiB',round(available_gib(),1),flush=True)
    if x.preflight_only:return
    if available_gib()<36:raise RuntimeError('Initial memory admission below 36GiB')
    model,tokenizer,model_receipt=prior_model.load_model('mopd_249')
    try:
        if model_receipt['revision']!=MODEL_REV:raise RuntimeError('Checkpoint identity mismatch')
        if model.config.eos_token_id!=1 or model.generation_config.eos_token_id!=1:
            raise RuntimeError('Unexpected model EOS config drift')
        tokenizer.padding_side='left'
        measured=[];marker_stats=defaultdict(list)
        for offset in range(0,len(cases),2):
            if available_gib()<22:raise RuntimeError('In-run memory safety stop')
            rows=cases[offset:offset+2]
            encoded=tokenizer([r['prompt'] for r in rows],padding=True,return_tensors='pt',
                              truncation=False)
            encoded.pop('token_type_ids',None)
            if encoded['input_ids'].shape[1]>1200:raise RuntimeError('Context token admission failed')
            with torch.inference_mode():
                output=model.generate(**encoded,do_sample=False,num_beams=1,
                    max_new_tokens=MAX_NEW,use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
            traces=output[:,encoded['input_ids'].shape[1]:]
            for row,tokens in zip(rows,traces):
                texts,stats=interpret(tokens.tolist(),tokenizer,row['context'],row['question'])
                marker_stats[row['position']].append(stats)
                for arm,text in texts.items():
                    metric=prev.score_one(row,text,bool(stats['cap']))
                    measured.append({'placement':row['position'],'strategy':arm,
                       'unanswerable':row['impossible'],'strict':metric['strict'],
                       'lenient':metric['lenient'],
                       'nonabstention':metric['nonabstain_impossible'],
                       'unexpected_unknown':metric['unexpected_unknown'],
                       'changed_from_native':int(text!=texts['native'])})
            if offset%8==0:
                print('EXP032B_REAL_GENERATION_PROGRESS',offset+len(rows),'/',len(cases),
                      'elapsed_sec',round(time.monotonic()-START,1),flush=True)
            del encoded,output,traces
        if len(measured)!=len(cases)*len(STRATEGIES):
            raise RuntimeError('Incomplete readout census')
        totals={place:{arm:summarize(measured,place,arm) for arm in STRATEGIES}
                for place in PLACES}
        if any(len(marker_stats[p])!=16 for p in PLACES):
            raise RuntimeError('Marker inventory incomplete')
        markdown=render(totals,marker_stats,receipt,model_receipt)
        pending=x.out.with_name(x.out.name+'.tmp')
        if pending.exists():raise RuntimeError('Scratch collision')
        pending.write_text(markdown,encoding='utf-8');pending.replace(x.out)
        print('EXP032B_COMPLETE',sha(x.out),'summary',totals,
              'elapsed_sec',round(time.monotonic()-START,1),flush=True)
    finally:
        del model
if __name__=='__main__':main()
