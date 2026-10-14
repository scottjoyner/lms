"""EXP032F: deterministic article-disjoint SQuAD 2.0 evaluation fixture.

Pure source filtering, no model behavior or gold-dependent parameter fitting.
"""
import hashlib
import json
import re
from pathlib import Path

N_TOPICS=16
EXISTING_N=8

def sha(data):return hashlib.sha256(data).hexdigest()

def select(prior):
    prev,_,_=prior.source_rows()
    original={x['topic_digest'] for x in prev}
    if len(original)!=EXISTING_N:raise RuntimeError('Unexpected 8-source ancestor')
    data=json.loads(prior.SOURCE.read_text())
    eligible=[]
    for topic in data['data']:
        title=topic['title']
        tag=sha(title.encode())[:16]
        if tag in original:continue
        for para in topic['paragraphs']:
            context=para['context']
            if not 190<len(context)<1800:continue
            a=[];u=[]
            for q in para['qas']:
                if not q['question'].strip().lower().startswith(('who ','where ','which ')):continue
                if q['is_impossible']:
                    if q['answers']:raise RuntimeError('Impossible human annotation with supplied answers')
                    u.append(q)
                elif q['answers'] and all(
                    re.fullmatch(r'[A-Za-z][A-Za-z\-]{1,25}',ans['text']) and
                    context[ans['answer_start']:ans['answer_start']+len(ans['text'])]==ans['text']
                    for ans in q['answers']):
                    a.append(q)
            if not a or not u:continue
            picka=min(a,key=lambda q:sha(q['id'].encode()))
            picku=min(u,key=lambda q:sha(q['id'].encode()))
            eligible.append((sha((title+context).encode()),title,tag,context,picka,picku))
    chosen=[];seen=set(original)
    for _,title,tag,context,a,u in sorted(eligible,key=lambda item:item[0]):
        if tag in seen:continue
        seen.add(tag)
        chosen.append((tag,context,a,u))
        if len(chosen)==N_TOPICS:break
    if len(chosen)!=N_TOPICS or len(set(tag for tag,_,_,_ in chosen))!=N_TOPICS:
        raise RuntimeError('Missing 16 article-disjoint source topics')
    rows=[]
    for i,(tag,context,a,u) in enumerate(chosen):
        for q in (a,u):
            case={'article_index':i,'topic_digest':tag,
                  'para_digest':sha(context.encode()),
                  'qa_id':q['id'],'context':context,'question':q['question'],
                  'impossible':bool(q['is_impossible']),
                  'answers':tuple(sorted({ans['text'] for ans in q['answers']})),
                  'position':'short'}
            case['prompt']=prior.prompt_for(case,'','short')
            rows.append(case)
    if len(rows)!=2*N_TOPICS or sum(x['impossible'] for x in rows)!=N_TOPICS:
        raise RuntimeError('Wrong human-annotated QA class balance')
    proof=sha(json.dumps([{'title_digest':r['topic_digest'],
           'para_digest':r['para_digest'],'qid':r['qa_id'],
           'impossible':r['impossible'],'answers':r['answers']}
           for r in rows],sort_keys=True,separators=(',',':')).encode())
    return rows,proof,len(eligible)
