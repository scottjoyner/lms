"""EXP027 source-disjoint authored synthetic directional QA fixture.

No published EXP015/019/021 names or relation families are reused. These
are algorithmically checkable labels, NOT independent human-adjudicated gold.
"""
import hashlib
import json
import random
import sys
from pathlib import Path

SEED=20261027
NAMES='''Arlowe Brisden Calwick Derwent Eastmere Farring Grendon Huxley
Isenford Jorvik Keldon Lonsdale Marden Norcroft Ormsby Prestwick
Quenby Renshaw Sedgewick Talmere Ulwick Varron Westmere Xandor
Yelden Zorwick Briarley Crampton Dryford Evermoor Frensley Grantham'''.split()
FAMILIES={
'audited':('{a} audited {b} at the museum.', 'Who conducted the audit?', 'Who was audited?'),
'mentored':('{a} mentored {b} during the apprenticeship.', 'Who provided mentoring?', 'Who was mentored?'),
'warned':('{a} warned {b} about the storm.', 'Who gave the warning?', 'Who received the warning?'),
'congratulated':('{a} congratulated {b} for the achievement.', 'Who offered congratulations?', 'Who received congratulations?'),
'examined':('{a} examined {b} during the assessment.', 'Who performed the examination?', 'Who was examined?'),
'hired':('{a} hired {b} for the position.', 'Who did the hiring?', 'Who was hired?'),
'questioned':('{a} questioned {b} during the hearing.', 'Who asked the questions?', 'Who was questioned?'),
'vaccinated':('{a} vaccinated {b} at the clinic.', 'Who administered the vaccine?', 'Who received the vaccine?'),
}

def check_disjoint():
    # Repo-local source files are part of reproducible model-worktree checkout.
    directory=Path('/home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026')
    sys.path.insert(0,str(directory))
    import exp015_fixture as f15
    import exp019_fixture as f19
    import exp021_fixture as f21
    for old in (f15,f19,f21):
        if set(NAMES)&set(old.NAMES):raise RuntimeError('EXP027 names overlapped prior corpus')
        if set(FAMILIES)&set(old.FAMILIES):raise RuntimeError('EXP027 family overlapped prior corpus')
    if len(set(NAMES))!=32 or len(FAMILIES)!=8:raise RuntimeError('Source cardinality')

def make():
    rng=random.Random(SEED)
    names=NAMES[:]
    rng.shuffle(names)
    rows=[]
    for g in range(16):
        family=sorted(FAMILIES)[g//2]
        frame,q_a,q_b=FAMILIES[family]
        name0,name1=names[2*g:2*g+2]
        for state in (0,1):
            actor,target=(name0,name1) if state==0 else (name1,name0)
            fact=frame.format(a=actor,b=target)
            neutral=f'{actor} and {target} attended the same orientation session.'
            for role,question,correct_name in (
                    ('actor',q_a,actor),('second_party',q_b,target)):
                for order in (0,1):
                    choice_a,choice_b=(name0,name1) if order==0 else (name1,name0)
                    gold='A' if choice_a==correct_name else 'B'
                    for condition,source in (('factual',fact),('neutral',neutral)):
                        prompt=('Use the supplied statement to answer the question. Reply only A or B.\n'
                            f'Statement: {source}\nQuestion: {question}\n'
                            f'Choice A: {choice_a}\nChoice B: {choice_b}\nAnswer:')
                        rows.append({'group':g,'family':family,'state':state,
                           'role':role,'order':order,'condition':condition,
                           'correct':gold,'prompt':prompt})
    if len(rows)!=256 or len({r['prompt'] for r in rows})!=256:
        raise RuntimeError('EXP027 256 unique stimulus count failed')
    return rows

def receipt(rows):
    if len(rows)!=256:raise RuntimeError('Wrong fixture size')
    data=json.dumps(rows,sort_keys=True,separators=(',',':')).encode()
    return {'sha256':hashlib.sha256(data).hexdigest(),
            'seed':SEED,'name_pairs':16,'families':8,
            'prompts':256,'factual':128,'neutral':128,
            'human_reviewed':False}
