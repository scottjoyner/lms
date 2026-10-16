"""EXP047 deterministic QASPER last-eight-paper decoding control, no model calls."""
import hashlib,json,re
from k2_qasper_evidence_fixture import candidates,build as old45,sha,score as base_score,prompt as base_prompt
from k2_qasper_retrieval_fixture import examples as old46
PAPERS=8
KINDS=('supported','human_unanswerable')
CAPS=(16,64)
MODES=('default','yarn')
def cohort():
    available=candidates()
    if len(available)<32:raise RuntimeError('HOLD insufficient eligible QASPER papers')
    blocked={x['pid'] for x in old45()}|{x['pid'] for x in old46()[0]}
    chosen=available[24:32]
    if len(chosen)!=8 or any(pid in blocked for _,pid,*_ in chosen):
        raise RuntimeError('HOLD EXP047 paper overlaps earlier studies')
    rows=[]
    for topic,(_,pid,p,answerable,unanswer,_,unanswer_evidence) in enumerate(chosen):
        qa,gold,evidence=answerable
        for kind,q,g,c in (('supported',qa,gold,evidence),('human_unanswerable',unanswer,'UNKNOWN',unanswer_evidence)):
            rows.append({'topic':topic,'pid':pid,'title':p['title'].strip(),'qid':q['question_id'],
                         'kind':kind,'question':q['question'].strip(),'gold':g,'context':c})
    if len(rows)!=16 or len({r['pid'] for r in rows})!=8:raise RuntimeError('HOLD cohort cardinality')
    if any(r['gold'] not in r['context'] for r in rows if r['kind']=='supported'):
        raise RuntimeError('HOLD missing oracle gold')
    return rows

def prompt(r):
    # Frozen similar semantics to EXP045, no prompt tuning from model outcomes.
    return base_prompt(r)

def receipt(rows=None):
    rows=cohort() if rows is None else rows
    metadata=[{'topic':r['topic'],'qid':r['qid'],'kind':r['kind'],
         'gold':r['gold'],'prompt':prompt(r)} for r in rows]
    return sha(json.dumps(metadata,ensure_ascii=False,sort_keys=True,separators=(',',':')))

def extract_span(s):
    """Single deterministic rule, BEFORE knowing gold: first line -> optional
    preamble removal -> first sentence/semicolon cutoff; no search for gold.
    Does NOT count a mere occurrence of gold in an arbitrary long output.
    """
    s=s.split('\n',1)[0].strip()
    s=re.sub(r'^(?:answer\s*:|a\s*:|the answer is\s+|it is\s+)', '',s,flags=re.I).strip()
    # Only break at punctuation with whitespace/EOS so internal decimal 3.14 survives.
    s=re.split(r'[.!?;](?=\s|$)',s,maxsplit=1)[0].strip()
    return s

def score(r,visible,hit_newline=False,cap=False,eos=False):
    strict=base_score(r,visible,hit_cap=cap,hit_newline=hit_newline)
    extracted=extract_span(visible)
    sec=base_score(r,extracted)
    strict['extracted_exact']=sec['strict']
    strict['extracted_f1']=sec['f1']
    strict['eos']=int(eos)
    return strict

def aggregate(rows,scores):
    if len(rows)!=16 or len(scores)!=16:raise RuntimeError('HOLD incomplete generations')
    out={}
    for kind in KINDS:
        chosen=[v for r,v in zip(rows,scores) if r['kind']==kind]
        if len(chosen)!=8:raise RuntimeError('HOLD wrong category denominator')
        d={'n':8}
        for field in ('strict','extracted_exact','unknown','newline','eos','cap','empty'):
            d[field]=sum(x[field] for x in chosen)
        for field in ('f1','extracted_f1'):
            d[field]=round(sum(x[field] for x in chosen)/8,6)
        if d['newline']+d['eos']+d['cap']!=8:raise RuntimeError('HOLD stop reasons')
        out[kind]=d
    return out
if __name__=='__main__':
 r=cohort()
 print('EXP047_PAIRS',len(r),'COHORT_SHA',receipt(r))
