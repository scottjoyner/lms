#!/usr/bin/env python3
"""EXP032D: frozen K2 first-word candidate, source gate, separate Qwen YES verifier.

Fail closed on any source/model/budget mismatch. Aggregate Markdown only.
"""
import argparse
import gc
import hashlib
import os
import re
import sys
import time
from pathlib import Path

for k,v in {'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
            'CUDA_VISIBLE_DEVICES':'-1','OPENBLAS_NUM_THREADS':'1',
            'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}.items():os.environ[k]=v
import torch

PROTOCOL='5746cc1608351b3158b89aed8c51a5f998f5fbb55de92dbb21b9698b0143f739'
K2_REV='7abae15437c2e7f1b2cd986be562344f204a17f3'
QWEN_REV='2fc06364715b967f1860aea9cf38778875588b17'
EXP032B_SHA='a4c23ec645a5dfe825257ce8ed481d068011c8ecc0fbb5c07597075364ba8a4c'
N=16
START=time.monotonic()
torch.set_num_threads(2)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def memory_gib():
    return next(int(x.split()[1]) for x in Path('/proc/meminfo').read_text().splitlines()
                if x.startswith('MemAvailable:'))/1024**2

def candidate_from_text(text):
    m=re.match(r'^\s*([A-Za-z][A-Za-z-]*)',text)
    return m.group(1) if m else ''

def word_in(text,word):
    if not word:return False
    return bool(re.search(r'(?<![A-Za-z])'+re.escape(word)+r'(?![A-Za-z])',text,re.I))

def candidate_rules(candidate,passage,question):
    # No human gold or answerability flag given to any rule.
    first=candidate if candidate else 'UNKNOWN'
    lexical=bool(candidate and 2<=len(candidate)<=26 and word_in(passage,candidate))
    exclude=bool(lexical and not word_in(question,candidate))
    return {'first_word':first,'lexical_gate':candidate if lexical else 'UNKNOWN',
            'question_exclusion':candidate if exclude else 'UNKNOWN',
            'verified_candidate_eligible':exclude}

def verdict_prompt(passage,question,candidate):
    assert candidate and '\n' not in candidate and '\r' not in candidate
    return ('Decide whether the proposed candidate is DIRECTLY supported by the passage '
            'as the answer to this PARTICULAR question. Do not answer the question '
            'yourself; only evaluate the proposed candidate. If the passage does not '
            'entail it, answer NO. Respond with exactly YES or NO.\n'
            f'Passage:\n{passage}\nQuestion: {question}\n'
            f'Proposed candidate: {candidate}\nVerdict:')

def normalized_verdict(text,cap):
    # STRICT entire output; malformed, overlong and cap are denied.
    if cap:return 'MALFORMED'
    m=re.fullmatch(r'\s*(YES|NO)\s*[.!]?\s*',text,re.I)
    return m.group(1).upper() if m else 'MALFORMED'

def stats(results,kind):
    x=[r for r in results if r['strategy']==kind]
    pos=[r for r in x if not r['unanswerable']]
    neg=[r for r in x if r['unanswerable']]
    if len(pos)!=8 or len(neg)!=8:raise RuntimeError('Human-gold score cohort incomplete')
    return {'correct':sum(r['strict'] for r in pos),
            'false_accept':sum(r['nonabstain_negative'] for r in neg),
            'abstain_negative':sum(r['strict'] for r in neg),
            'missed_positive':sum(r['abstain_positive'] for r in pos)}

def report(scored,decisions,custody,selection,base_prompt):
    lines=['# EXP-032D — Deny-only Qwen candidate judge for K2 human-gold QA',
      '', '**2026-10-14 | real two-frozen-model CPU witness, 16 SQuAD2 dev cases.**',
      'K2 first-word extraction supplies proposed answers; Qwen3.5-0.8B',
      'judges direct support by passage and question in its native chat format.',
      'All gold scoring AFTER judge decision; gold never supplied to the judge.',
      'Original human annotations: SQuAD2 development, Rajpurkar et al. (2018),',
      'CC BY-SA 4.0; https://rajpurkar.github.io/SQuAD-explorer/',
      '', '## Source, model and preregistration receipts','',
      f'EXP032D preregistration SHA256 `{PROTOCOL}`.',
      f'Original EXP032B measured stop/lexical-gate SHA256 `{EXP032B_SHA}`.',
      f'Original human-gold selection SHA256 `{selection}`.',
      f'Original K2 prompt/placement SHA256 `{base_prompt}`.',
      f"K2 official mOPD revision `{custody['k2_revision']}`; config SHA256 `{custody['k2_config']}`.",
      f"Qwen official 0.8B revision `{custody['qwen_revision']}`; config SHA256 `{custody['qwen_config']}`.",
      f"Qwen native chat template: **{custody['qwen_chat']}**; no invented K2 chat template.",
      'Original K2 native EOS, greedy 32 tokens; Qwen strict YES/NO',
      'native chat, greedy 16 tokens. Models loaded sequentially.',
      '', '## Human-annotated gold and deny-only decision behavior','',
      'Eight human-answerable + eight human-unanswerable cases, short source only.',
      '| Rule | Strict correct /8 answerable | Unsupported non-UNKNOWN /8 unanswerable | Exact UNKNOWN /8 unanswerable | Missed UNKNOWN /8 answerable |',
      '|---|---:|---:|---:|---:|']
    for arm in ('first_word','lexical_gate','question_exclusion','qwen_verifier','always_unknown'):
        s=stats(scored,arm)
        lines.append(f"| {arm} | {s['correct']} | {s['false_accept']} | {s['abstain_negative']} | {s['missed_positive']} |")
    lines+=['','## Independent verifier response contract','',
      f"Actual candidate evaluations requested of frozen Qwen: **{decisions['asked']}**.",
      f"Strict whole-output YES: **{decisions['yes']}**.",
      f"Strict whole-output NO: **{decisions['no']}**.",
      f"Malformed, inconclusive or output cap denied: **{decisions['malformed']}**.",
      f"Qwen 16-token caps (denied): **{decisions['cap']}**.",
      'All cases not eligible for question-exclusion lexical gate were',
      'denied BEFORE Qwen; a model YES alone never bypasses lexical policy.',
      '', '## Interpretation constraints','',
      'Qwen is another trained model, NOT an independently governed, calibrated',
      'or reliable evidence verifier. It may share SQuAD2 training exposure.',
      'The public dev subset has just eight article pairs already used in',
      'earlier K2 trials; these observations are exploratory and are not',
      'blind independent validation or a population reliability measure.',
      'Strict YES/NO full-output parsing is fail-closed; a label other than',
      'exact YES is denied, which can sacrifice legitimate answers.',
      'The model has not been tested against prompt injection, adversarial',
      'contradictions, unseen domains, calibration drift or larger cohorts.',
      'No production trust, router promotion, model training, NAS, graph',
      'or service changes have been authorized or performed.',
      '', '## Reproduction','',
      '`python research/k2_qwen_candidate_guard.py --evidence-dir /home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026 --out /home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026/EXPERIMENT-032D-QWEN-VERIFIER.md`','']
    return '\n'.join(lines)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--preflight-only',action='store_true')
    a=p.parse_args()
    if a.out.exists():raise SystemExit('STOP: measured report exists')
    if sha(a.evidence_dir/'EXPERIMENT-032D-QWEN-VERIFIER-PROTOCOL.md')!=PROTOCOL:
        raise RuntimeError('Pre-inference EXP032D protocol mismatch')
    if sha(a.evidence_dir/'EXPERIMENT-032B-STOP-POSTHOC.md')!=EXP032B_SHA:
        raise RuntimeError('Measured EXP032B source changed')
    sys.path.insert(0,str(a.evidence_dir))
    import k2_human_grounded_qa as prior
    import run_exp021_free_response as loaders
    from transformers import AutoTokenizer
    import run_exp018_k2_intermediate_gates as e18
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    import k2_posthoc_boundary as boundary
    tok=AutoTokenizer.from_pretrained(str(e18.stage_path('mopd_249')),
             local_files_only=True,trust_remote_code=False)
    rows,source_receipt=boundary.preflight(a.evidence_dir,prior,tok)
    rows=[row for row in rows if row['position']=='short']
    if len(rows)!=16 or sum(r['impossible'] for r in rows)!=8:
        raise RuntimeError('Wrong source cohort')
    print('EXP032D_PREFLIGHT',source_receipt,'n',len(rows),
          'memory_gib',round(memory_gib(),1),flush=True)
    if a.preflight_only:return
    if memory_gib()<36:raise RuntimeError('Memory gate pre K2')
    model,tok,k2_custody=loaders.load_model('mopd_249')
    if k2_custody['revision']!=K2_REV:
        raise RuntimeError('K2 revision drift')
    tok.padding_side='left';candidates=[];original=[]
    try:
        for i in range(0,len(rows),2):
            if memory_gib()<22:raise RuntimeError('K2 in-run memory gate')
            batch=rows[i:i+2]
            tokens=tok([r['prompt'] for r in batch],padding=True,
                       return_tensors='pt',truncation=False)
            tokens.pop('token_type_ids',None)
            with torch.inference_mode():
                out=model.generate(**tokens,max_new_tokens=32,do_sample=False,
                     num_beams=1,use_cache=True,eos_token_id=tok.eos_token_id,
                     pad_token_id=tok.pad_token_id)
            for row,ids in zip(batch,out[:,tokens['input_ids'].shape[1]:]):
                ids=ids.tolist()
                end=next((j for j,v in enumerate(ids) if v==tok.eos_token_id),len(ids))
                text=tok.decode(ids[:end],skip_special_tokens=True)
                original.append(prior.score_one(row,text,end>=32))
                cand=candidate_from_text(text)
                candidates.append(candidate_rules(cand,row['context'],row['question']))
            if i%4==0:print('EXP032D_K2_GENERATED',i+len(batch),'/',len(rows),flush=True)
            del tokens,out
    finally:
        del model
        gc.collect()
    if len(original)!=16 or len(candidates)!=16:
        raise RuntimeError('Incomplete K2 generation')
    if (sum(x['strict'] for x in original)!=0 or
        sum(x['lenient'] for x in original if not x['impossible'])!=7 or
        sum(x['hit_cap'] for x in original)!=9):
        raise RuntimeError('Original K2 EXP029 replay deviates')
    pending=[i for i,v in enumerate(candidates) if v['verified_candidate_eligible']]
    if not pending:raise RuntimeError('No valid candidate evaluation cases')
    if memory_gib()<36:raise RuntimeError('Memory gate before Qwen model')
    qwen,qtokenizer,qwen_custody=loaders.load_model('qwen08_chat')
    if qwen_custody['revision']!=QWEN_REV or not qwen_custody['native_chat_template']:
        raise RuntimeError('Qwen published revision/template mismatch')
    qtokenizer.padding_side='left'
    verdicts={};cap_count=0
    try:
        for k in range(0,len(pending),2):
            if memory_gib()<22:raise RuntimeError('Qwen in-run memory gate')
            indices=pending[k:k+2]
            queries=[verdict_prompt(rows[i]['context'],rows[i]['question'],
                    candidates[i]['question_exclusion']) for i in indices]
            formatted=[qtokenizer.apply_chat_template([{'role':'user','content':q}],
                       add_generation_prompt=True,tokenize=False) for q in queries]
            inputs=qtokenizer(formatted,padding=True,return_tensors='pt',
                             truncation=False,add_special_tokens=False)
            inputs.pop('token_type_ids',None)
            if inputs['input_ids'].shape[1]>1400:raise RuntimeError('Verifier context too large')
            with torch.inference_mode():
                output=qwen.generate(**inputs,do_sample=False,num_beams=1,
                    max_new_tokens=16,use_cache=True,
                    eos_token_id=qtokenizer.eos_token_id,
                    pad_token_id=qtokenizer.pad_token_id)
            for row_idx,ids in zip(indices,output[:,inputs['input_ids'].shape[1]:]):
                ids=ids.tolist();loc=next((j for j,v in enumerate(ids) if v==qtokenizer.eos_token_id),len(ids))
                cap=(loc>=16)
                decoded=qtokenizer.decode(ids[:loc],skip_special_tokens=True)
                verdicts[row_idx]=normalized_verdict(decoded,cap)
                cap_count+=int(cap)
            print('EXP032D_QWEN_VERIFIED',min(k+len(indices),len(pending)),'/',len(pending),flush=True)
            del inputs,output
    finally:
        del qwen;gc.collect()
    if len(verdicts)!=len(pending):raise RuntimeError('Incomplete Qwen verdict inventory')
    measured=[]
    for i,row in enumerate(rows):
        policy=candidates[i]
        choices={'first_word':policy['first_word'],
            'lexical_gate':policy['lexical_gate'],
            'question_exclusion':policy['question_exclusion'],
            'qwen_verifier':policy['question_exclusion'] if verdicts.get(i)=='YES' else 'UNKNOWN',
            'always_unknown':'UNKNOWN'}
        for arm,answer in choices.items():
            m=prior.score_one(row,answer,False)
            measured.append({'strategy':arm,'unanswerable':row['impossible'],
                'strict':m['strict'],'nonabstain_negative':m['nonabstain_impossible'],
                'abstain_positive':m['unexpected_unknown']})
    decisions={'asked':len(verdicts),'yes':sum(v=='YES' for v in verdicts.values()),
               'no':sum(v=='NO' for v in verdicts.values()),
               'malformed':sum(v=='MALFORMED' for v in verdicts.values()),
               'cap':cap_count}
    assert decisions['yes']+decisions['no']+decisions['malformed']==decisions['asked']
    custody={'k2_revision':k2_custody['revision'],
             'k2_config':k2_custody['model_config_sha256'],
             'qwen_revision':qwen_custody['revision'],
             'qwen_config':qwen_custody['model_config_sha256'],
             'qwen_chat':qwen_custody['native_chat_template']}
    doc=report(measured,decisions,custody,source_receipt,boundary.EXP029_PROMPTS)
    tmp=a.out.with_name(a.out.name+'.tmp')
    if tmp.exists():raise RuntimeError('Scratch file collision')
    tmp.write_text(doc,encoding='utf8');tmp.replace(a.out)
    print('EXP032D_COMPLETE',sha(a.out),'verdicts',decisions,
          'scores',{k:stats(measured,k) for k in ('first_word','lexical_gate','question_exclusion','qwen_verifier','always_unknown')},
          'elapsed_s',round(time.monotonic()-START,1))

if __name__=='__main__':main()
