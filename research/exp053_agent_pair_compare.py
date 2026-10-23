#!/usr/bin/env python3
"""EXP-053 exact-source Qwen / MiniCPM machine-proxy paired comparison.

Exploratory, not human adjudication. Only immutable private case receipts are read;
no checkpoint loading, licensed source redistribution or model inference.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

from exp053_agent_qwen_cpu import (FROZEN, RECEIPTS as QROOT, private_dir,
                                   read_receipt, verify_case as verify_q, sha,
                                   create_immutable)
from exp053_agent_minicpm_cpu import (RECEIPTS as MROOT,
                                      verify_case as verify_m)
from exp053_agent_qwen_summary import case_margins, source_data

METRICS = ('D','N','S')
SEED = 53053
BOOTSTRAP = 2000


def check(ok, why):
    if not ok: raise ValueError('HOLD EXP053 MACHINE PAIR: '+why)


def case_compare(q, m):
    check(q['case_id'] == m['case_id'], 'unpaired article identity')
    qrows={(x['arm'],x['rotation']):x for x in q['conditions']}
    mrows={(x['arm'],x['rotation']):x for x in m['conditions']}
    check(set(qrows)==set(mrows) and len(qrows)==6, 'missing source control')
    for key in qrows:
        x,y=qrows[key],mrows[key]
        check(x['prompt_sha256']==y['prompt_sha256'], 'source prompt hash differs')
        check(x['gold_label']==y['gold_label'] and x['foil_label']==y['foil_label'], 'answer label mismatch')
    return case_margins(q),case_margins(m)


def summarize_pairs(pairs):
    n=len(pairs)
    check(1<=n<=64, 'cohort size invalid')
    ans={}
    for key in METRICS:
        qvals=[q[key] for q,m in pairs];mvals=[m[key] for q,m in pairs]
        delta=[x-y for x,y in zip(qvals,mvals)]
        check(all(math.isfinite(z) for z in qvals+mvals+delta),'invalid finite value')
        rng=random.Random(SEED+ord(key))
        draws=sorted(statistics.mean(delta[rng.randrange(n)] for _ in range(n)) for _ in range(BOOTSTRAP))
        ans[key]={'qwen_mean':round(statistics.mean(qvals),6),
                  'minicpm_mean':round(statistics.mean(mvals),6),
                  'paired_qwen_minus_minicpm_mean':round(statistics.mean(delta),6),
                  'paired_bootstrap_95pct_descriptive':[
                      round(draws[int(BOOTSTRAP*.025)],6),
                      round(draws[int(BOOTSTRAP*.975)],6)],
                  'positive_qwen':sum(v>0 for v in qvals),
                  'positive_minicpm':sum(v>0 for v in mvals),
                  'paired_qwen_greater':sum(v>0 for v in delta),
                  'n_articles':n}
    return ans


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--count',type=int,default=64)
    ap.add_argument('--write-private',action='store_true')
    a=ap.parse_args()
    check(a.count==64,'full frozen cohort required for paired comparison')
    private_dir(QROOT);private_dir(MROOT)
    cohort,cases,qcheck=source_data()
    mcheck=json.loads(FROZEN.read_bytes())['models']['MiniCPM-V-4.7-1B']
    qsha=sha(Path(__file__).with_name('exp053_agent_qwen_cpu.py').read_bytes())
    msha=sha(Path(__file__).with_name('exp053_agent_minicpm_cpu.py').read_bytes())
    pairs=[];ids=[];case_hashes=[]
    for i,(cid,_) in enumerate(cases):
        qfile=QROOT/('case-%03d.json'%i)
        mfile=MROOT/('case-%03d.json'%i)
        q=read_receipt(qfile);m=read_receipt(mfile)
        verify_q(q,i,cid,qcheck['checkpoint_receipt_sha256'],cohort['packet_sha256'],qsha)
        verify_m(m,i,cid,mcheck['checkpoint_receipt_sha256'],cohort['packet_sha256'],msha)
        check(q.get('condition_count')==6 and m.get('condition_count')==6, 'condition count')
        check(q.get('tokenizer_sha256')==qcheck['tokenizer_sha256'] and m.get('tokenizer_sha256')==mcheck['tokenizer_sha256'], 'tokenizer receipts')
        pairs.append(case_compare(q,m))
        ids.append(cid)
        case_hashes.append({'index':i,'qwen_sha256':hashlib.sha256(qfile.read_bytes()).hexdigest(),
                            'minicpm_sha256':hashlib.sha256(mfile.read_bytes()).hexdigest()})
    check(len(set(ids))==64,'source distinctness failure')
    check(not list(QROOT.glob('failure-*.json')) and not list(MROOT.glob('failure-*.json')),'unreconciled failure records')
    result={'schema':'exp053-machine-proxy-two-model-paired-v1',
            'complete_two_models':True,'all_three_models_complete':False,
            'human_adjudicated':False,'causal_grounding_proven':False,
            'case_count':64,'source_rotations_per_model':384,
            'common_source_prompt_sha256_alignment':True,
            'qwen_checkpoint_receipt_sha256':qcheck['checkpoint_receipt_sha256'],
            'minicpm_checkpoint_receipt_sha256':mcheck['checkpoint_receipt_sha256'],
            'qwen_runner_sha256':qsha,'minicpm_runner_sha256':msha,
            'paired_per_article_machine_proxy':summarize_pairs(pairs),
            'case_receipt_hashes':case_hashes,
            'limit':'Descriptive paired bootstrap on same frozen machine exam-answer proxy cohort; not preregistered human-adjudicated experiment, native modality parity, no contamination claim.'}
    if not a.write_private:
        print('EXP053_PAIR_PREFLIGHT_OK articles=64 per_model_forwards=384; private_output=none',flush=True)
        print('EXP053_PAIR_DESCRIPTIVE',result['paired_per_article_machine_proxy'],flush=True)
        return
    receipt_sha=create_immutable(MROOT,'paired-qwen-minicpm-64case-v1.json',result)
    print('EXP053_PAIR_PRIVATE_COMPLETE articles=64 two_models=2 pair_sha256='+receipt_sha,flush=True)
    print('EXP053_PAIR_DESCRIPTIVE',result['paired_per_article_machine_proxy'],flush=True)

if __name__=='__main__':
    main()
