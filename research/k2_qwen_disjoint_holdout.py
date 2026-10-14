#!/usr/bin/env python3
"""EXP032F: same frozen K2→Qwen verification on 16 NEW article topics.

Human-labeled published SQuAD dev source, only aggregate Markdown retained.
"""
import argparse
import gc
import hashlib
import os
import sys
import time
from pathlib import Path
for k,v in {'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
            'CUDA_VISIBLE_DEVICES':'-1','OPENBLAS_NUM_THREADS':'1',
            'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}.items():os.environ[k]=v
import torch
import k2_qwen_candidate_guard as parent
import k2_disjoint_squad_fixture as fixture

PROTOCOL='937d28cc94374125952afcf07b9ab291f94bde223566d8217404efcf93ad3c33'
SOURCE_SELECTION='1382fc12d772628b8dfd2cf250dc36c8f35b9215d774db89dba05bc912e5ee65'
N_SOURCE=16
ARMS=('first_word','lexical_gate','question_exclusion','qwen_verifier','always_unknown')
START=time.monotonic()
torch.set_num_threads(2)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def summary(results,arm):
    score=[x for x in results if x['strategy']==arm]
    a=[x for x in score if not x['impossible']];u=[x for x in score if x['impossible']]
    if len(a)!=N_SOURCE or len(u)!=N_SOURCE:raise RuntimeError('Incorrect human annotated balance')
    return {'correct':sum(x['exact'] for x in a),
            'incorrect_accepted':sum(x['accepted'] and not x['exact'] for x in a),
            'negative_false_accept':sum(x['nonabstain_negative'] for x in u),
            'negative_unknown':sum(x['exact'] for x in u),
            'positive_unknown':sum(x['positive_unknown'] for x in a)}

def render(source_sha,custody,counts,candidate_count,caps,cases):
    lines=['# EXP-032F — Source-article-disjoint K2-to-Qwen verifier evaluation',
      '', '**2026-10-14 | actual frozen two-model 32-item CPU evaluation, research-only.**',
      'Source: official public SQuAD 2.0 dev (Rajpurkar et al. 2018), CC BY-SA 4.0,',
      'https://rajpurkar.github.io/SQuAD-explorer/. Sixteen article titles are',
      'disjoint from the eight original EXP029–032E source articles;',
      'public dev source may still be in model training, and this is NOT blind.',
      '', '## Exact prospective source and frozen checkpoint receipts','',
      f'EXP032F pre-inference protocol SHA256 `{PROTOCOL}`.',
      f'16-topic, 32-question fixture SHA256 `{source_sha}`.',
      'Original 8-article control source remains excluded by article hash.',
      f"K2 frozen official mOPD revision `{custody['k2_rev']}`; config SHA256 `{custody['k2_sha']}`.",
      f"Qwen frozen native-chat 0.8B revision `{custody['qwen_rev']}`; config SHA256 `{custody['qwen_sha']}`.",
      f"Qwen native chat template verified: **{custody['qwen_chat']}**.",
      'K2 plain 32-token greedy; Qwen strict whole-output YES/NO and EOS',
      'within 16 generated tokens; models separately resident on same CPU host.',
      '', '## Human-gold safety and answer recovery','',
      'Each arm: sixteen human-annotated answerable and sixteen',
      'human-unanswerable questions, one pair per disjoint source article.',
      '| Answer decision rule | Correct /16 answerable | Incorrect accepted /16 answerable | Unsupported non-UNKNOWN /16 unanswerable | Strict UNKNOWN /16 negatives | Missed UNKNOWN /16 positives |',
      '|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        x=counts[arm]
        lines.append(f"| {arm} | {x['correct']} | {x['incorrect_accepted']} | {x['negative_false_accept']} | {x['negative_unknown']} | {x['positive_unknown']} |")
    lines+=['','## Raw decoder and Qwen verdict custody','',
        f'Native K2 32-token cap events: **{caps["k2"]}/32**.',
        f'Eligible candidate judgments sent to frozen Qwen: **{candidate_count}/32**.',
        f'Exact Qwen YES: **{caps["yes"]}**; NO: **{caps["no"]}**;',
        f'malformed/ambiguous: **{caps["malformed"]}**;',
        f'token caps strictly denied: **{caps["qwen_cap"]}**.',
        '', '## Interpretation and safety decision','',
        'Model candidates and Qwen verifier decisions were produced without',
        'exposure to human gold or is_impossible labels; only subsequent',
        'scoring uses SQuAD2 annotated answerability. The source-domain',
        'disjointness is relative ONLY to our previous eight evaluated articles.',
        'It does NOT mean Qwen has never seen these public dev examples.',
        'First-word extraction, substring source matching and a separate',
        'Qwen YES verdict are not an independently governed semantic proof.',
        'False acceptance on human-unanswerable questions is a safety failure',
        'for the intended grounded-QA task, not necessarily a fabricated fact.',
        'Any apparent success is limited by public-source leakage, small',
        'article count, mechanical one-word selection and dependent pairs.',
        'A zero in sixteen would NOT establish production-ready abstention.',
        '**NO-GO** remains for unattended grounded K2 QA, routing, deployment',
        'or model promotion without independent source-disjoint adjudication.',
        'No raw passages, answers, model generated strings or per-item traces',
        'were stored; no NAS, graph, training, credentials, registry or live',
        'production effects.',
        '', '## Reproduce','',
        'Source generator: `research/k2_disjoint_squad_fixture.py`;',
        'frozen pipeline: `research/k2_qwen_disjoint_holdout.py`;',
        'tests: `tests/test_k2_qwen_disjoint_holdout.py`.',
        '`python research/k2_qwen_disjoint_holdout.py --evidence-dir /home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026 --out /home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026/EXPERIMENT-032F-ARTICLE-DISJOINT.md`','']
    return '\n'.join(lines)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--preflight-only',action='store_true')
    a=p.parse_args()
    if a.out.exists():raise SystemExit('STOP existing report target')
    if sha(a.evidence_dir/'EXPERIMENT-032F-ARTICLE-DISJOINT-PROTOCOL.md')!=PROTOCOL:
        raise RuntimeError('Prospective source protocol changed')
    sys.path.insert(0,str(a.evidence_dir))
    import k2_human_grounded_qa as qa
    import run_exp021_free_response as loaders
    source,receipt,eligible=fixture.select(qa)
    if receipt!=SOURCE_SELECTION or len(source)!=32:
        raise RuntimeError('Source article-disjoint fixture digest mismatch')
    print('EXP032F_SOURCE_PREFLIGHT',receipt,'eligible_source_pairs',eligible,
          'memory_gib',round(parent.memory_gib(),1),flush=True)
    if a.preflight_only:return
    if parent.memory_gib()<36:raise RuntimeError('Pre-K2 memory admission gate')
    model,tok,k2=loaders.load_model('mopd_249')
    if k2['revision']!=parent.K2_REV:raise RuntimeError('K2 official revision mismatch')
    tok.padding_side='left'
    # Preserve exact prompt-length metadata required by the frozen EXP029
    # scorer without adding it to the source-selection digest.
    for case in source:
        case['prompt_tokens']=len(tok.encode(case['prompt'],add_special_tokens=True))
    candidates=[];native=[]
    try:
        for i in range(0,len(source),2):
            if parent.memory_gib()<22:raise RuntimeError('K2 batch memory stop')
            sub=source[i:i+2]
            x=tok([r['prompt'] for r in sub],padding=True,return_tensors='pt',
                  truncation=False)
            x.pop('token_type_ids',None)
            if x['input_ids'].shape[1]>1100:raise RuntimeError('K2 source context too long')
            with torch.inference_mode():
                y=model.generate(**x,max_new_tokens=32,do_sample=False,num_beams=1,
                    use_cache=True,eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id)
            for row,ids in zip(sub,y[:,x['input_ids'].shape[1]:]):
                arr=ids.tolist();stop=next((j for j,v in enumerate(arr) if v==tok.eos_token_id),len(arr))
                decoded=tok.decode(arr[:stop],skip_special_tokens=True)
                candidate=parent.candidate_from_text(decoded)
                candidates.append(parent.candidate_rules(candidate,row['context'],row['question']))
                native.append(int(stop>=32))
            if i%8==0:print('EXP032F_K2_PROGRESS',i+len(sub),'/',len(source),flush=True)
            del x,y
    finally:
        del model;gc.collect()
    if len(candidates)!=32:raise RuntimeError('K2 native cohort incomplete')
    subset=[i for i,c in enumerate(candidates) if c['verified_candidate_eligible']]
    if not subset:raise RuntimeError('No Qwen-eligible candidate: unresolved low coverage')
    if parent.memory_gib()<34:raise RuntimeError('Amended pre-Qwen memory admission gate (34GiB)')
    qwen,qtok,qreceipt=loaders.load_model('qwen08_chat')
    if qreceipt['revision']!=parent.QWEN_REV or not qreceipt['native_chat_template']:
        raise RuntimeError('Qwen revision or native chat template mismatch')
    qtok.padding_side='left';verdicts={};caps=0
    try:
        for start in range(0,len(subset),2):
            if parent.memory_gib()<22:raise RuntimeError('Qwen batch memory stop')
            selected=subset[start:start+2]
            prompt=[parent.verdict_prompt(source[i]['context'],source[i]['question'],
                    candidates[i]['question_exclusion']) for i in selected]
            native_chat=[qtok.apply_chat_template([{'role':'user','content':q}],
                         add_generation_prompt=True,tokenize=False) for q in prompt]
            x=qtok(native_chat,padding=True,return_tensors='pt',
                   truncation=False,add_special_tokens=False)
            x.pop('token_type_ids',None)
            if x['input_ids'].shape[1]>1400:raise RuntimeError('Qwen context too long')
            with torch.inference_mode():
                y=qwen.generate(**x,max_new_tokens=16,do_sample=False,num_beams=1,
                    use_cache=True,eos_token_id=qtok.eos_token_id,pad_token_id=qtok.pad_token_id)
            for index,ids in zip(selected,y[:,x['input_ids'].shape[1]:]):
                ids=ids.tolist();stop=next((j for j,v in enumerate(ids) if v==qtok.eos_token_id),len(ids))
                capped=(stop>=16)
                output=qtok.decode(ids[:stop],skip_special_tokens=True)
                verdicts[index]=parent.normalized_verdict(output,capped)
                caps+=int(capped)
            if start%8==0:print('EXP032F_QWEN_PROGRESS',min(start+len(selected),len(subset)),'/',len(subset),flush=True)
            del x,y
    finally:
        del qwen;gc.collect()
    if len(verdicts)!=len(subset):raise RuntimeError('Qwen did not score all eligible candidates')
    measured=[]
    for i,row in enumerate(source):
        base=candidates[i]
        choices={'first_word':base['first_word'],
                 'lexical_gate':base['lexical_gate'],
                 'question_exclusion':base['question_exclusion'],
                 'qwen_verifier':base['question_exclusion'] if verdicts.get(i)=='YES' else 'UNKNOWN',
                 'always_unknown':'UNKNOWN'}
        for arm,out in choices.items():
            value=qa.score_one(row,out,False)
            measured.append({'strategy':arm,'impossible':row['impossible'],
                'exact':value['strict'],'accepted':out.strip().upper()!='UNKNOWN',
                'nonabstain_negative':value['nonabstain_impossible'],
                'positive_unknown':value['unexpected_unknown']})
    stats={arm:summary(measured,arm) for arm in ARMS}
    counts={'k2':sum(native),'yes':sum(v=='YES' for v in verdicts.values()),
            'no':sum(v=='NO' for v in verdicts.values()),
            'malformed':sum(v=='MALFORMED' for v in verdicts.values()),'qwen_cap':caps}
    if counts['yes']+counts['no']+counts['malformed']!=len(subset):
        raise RuntimeError('Verifier census fails closed')
    custody={'k2_rev':k2['revision'],'k2_sha':k2['model_config_sha256'],
             'qwen_rev':qreceipt['revision'],'qwen_sha':qreceipt['model_config_sha256'],
             'qwen_chat':qreceipt['native_chat_template']}
    doc=render(receipt,custody,stats,len(subset),counts,source)
    temp=a.out.with_name(a.out.name+'.tmp')
    if temp.exists():raise RuntimeError('Scratch path already exists')
    temp.write_text(doc,encoding='utf8');temp.replace(a.out)
    print('EXP032F_COMPLETE',sha(a.out),'cohort',len(source),'qwen',len(subset),
          'scores',stats,'verdicts',counts,'elapsed_s',round(time.monotonic()-START,1),flush=True)

if __name__=='__main__':main()
