#!/usr/bin/env python3
"""EXP029: pinned human-annotated SQuAD2 direct-generation, abstention and position test.

No raw dataset/predictions committed; Markdown-only aggregate observations.
"""
import argparse
import gc
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

for k,v in {'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','CUDA_VISIBLE_DEVICES':'-1',
            'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}.items():os.environ[k]=v
import torch

SOURCE=Path('/home/scott/git/.latent-squad2-dev-research/dev-v2.0.json')
SOURCE_SHA='80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8'
PROTOCOL_SHA='e6e7e2da0a1c58b15587384f77ee8d0c2c8760e5a105d099a154d411ec0b99ef'
FILL_TARGET=768
POSITIONS=('short','prefix','gap')
BATCH=2
MAX_NEW=32
STAGES=('mopd_249','qwen08_chat')
QUESTION_PREFIX=('who ','where ','which ')
START=time.monotonic()
torch.set_num_threads(2)

def sha_bytes(x):return hashlib.sha256(x).hexdigest()
def sha(path):return sha_bytes(Path(path).read_bytes())
def ram():
    return next(int(s.split()[1]) for s in Path('/proc/meminfo').read_text().splitlines()
                if s.startswith('MemAvailable:'))/1024**2

def source_rows(path=SOURCE):
    if sha(path)!=SOURCE_SHA:raise RuntimeError('STOP: SQuAD2 dev file checksum mismatch')
    source=json.loads(path.read_text())
    if source['version']!='v2.0':raise RuntimeError('Wrong QA source version')
    eligible=[]
    for section in source['data']:
        for paragraph in section['paragraphs']:
            context=paragraph['context']
            if not 190<len(context)<1800:continue
            answerable=[];impossible=[]
            for q in paragraph['qas']:
                if not q['question'].strip().lower().startswith(QUESTION_PREFIX):continue
                if q['is_impossible']:
                    if q['answers']:raise RuntimeError('Impossible QA unexpected gold')
                    impossible.append(q)
                elif (q['answers'] and all(
                    re.fullmatch(r'[A-Za-z][A-Za-z\-]{1,25}',a['text'])
                    and context[a['answer_start']:a['answer_start']+len(a['text'])]==a['text']
                    for a in q['answers'])):
                    answerable.append(q)
            if answerable and impossible:
                a=min(answerable,key=lambda q:sha_bytes(q['id'].encode()))
                b=min(impossible,key=lambda q:sha_bytes(q['id'].encode()))
                eligible.append((sha_bytes((section['title']+context).encode()),section['title'],context,a,b))
    selected=[];titles=set()
    for digest,title,context,a,b in sorted(eligible,key=lambda x:x[0]):
        if title in titles:continue
        titles.add(title)
        selected.append({'topic_digest':sha_bytes(title.encode())[:16],
                         'paragraph_digest':sha_bytes(context.encode()),
                         'context':context,'qas':(a,b)})
        if len(selected)==8:break
    if len(selected)!=8 or len(titles)!=8:raise RuntimeError('No 8 separate SQuAD2 topics')
    cases=[]
    for idx,item in enumerate(selected):
        for q in item['qas']:
            cases.append({'article_index':idx,'topic_digest':item['topic_digest'],
                          'para_digest':item['paragraph_digest'],'qa_id':q['id'],
                          'context':item['context'],'question':q['question'],
                          'impossible':bool(q['is_impossible']),
                          'answers':tuple(sorted({a['text'] for a in q['answers']}))})
    if len(cases)!=16 or sum(x['impossible'] for x in cases)!=8:raise RuntimeError('Wrong 8+8 QA balance')
    proof=sha_bytes(json.dumps([{'article':r['topic_digest'],'paragraph':r['para_digest'],
         'qid':r['qa_id'],'answers':r['answers'],'impossible':r['impossible']}
         for r in cases],sort_keys=True,separators=(',',':')).encode())
    return cases,proof,len(eligible)

def filler_text(tokenizer,cases):
    text=''.join(f'Neutral ledger line {i:04d} mentions an empty box on a shelf.\n'
                 for i in range(200))
    ids=tokenizer.encode(text,add_special_tokens=False)
    if len(ids)<FILL_TARGET+16:raise RuntimeError('Filler too short')
    filler=tokenizer.decode(ids[:FILL_TARGET],skip_special_tokens=True)
    n=len(tokenizer.encode(filler,add_special_tokens=False))
    if abs(n-FILL_TARGET)>3:raise RuntimeError('Filler token count deviates >3')
    # Prevent a correct short answer being spuriously supplied in the filler.
    words=set(re.findall(r'[a-z]+',filler.lower()))
    for case in cases:
        for answer in case['answers']:
            if answer.casefold() in words:raise RuntimeError('Gold answer leaked into filler '+str(case['article_index']))
    return filler,n

def prompt_for(case,filler,position):
    instruction=('Answer the question using only the passage. '
                 'Give the shortest answer from the passage. '
                 'If the passage does not answer the question, respond exactly UNKNOWN.\n')
    content=f"Passage:\n{case['context']}\n"
    question=f"Question: {case['question']}\nAnswer:"
    if position=='short':return instruction+content+question
    if position=='prefix':return instruction+filler+'\n'+content+question
    if position=='gap':return instruction+content+filler+'\n'+question
    raise ValueError('Unknown position')

def generate_rows(tokenizer,cases,stage):
    filler,filler_tokens=filler_text(tokenizer,cases)
    all_rows=[]
    for position in POSITIONS:
        for c in cases:
            prompt=prompt_for(c,filler,position)
            formatted=(tokenizer.apply_chat_template([{'role':'user','content':prompt}],
                 tokenize=False,add_generation_prompt=True) if stage=='qwen08_chat' else prompt)
            tokens=len(tokenizer.encode(formatted,add_special_tokens=stage!='qwen08_chat'))
            gap_str=formatted[:formatted.rfind('Question: ')]
            passage=formatted[formatted.index('Passage:\n'):]
            question=formatted[formatted.rfind('Question: '):]
            # Prefix/Gap token counts differ slightly under subword tokenization,
            # but comparison is per same source question, model & chat wrapper.
            all_rows.append({**c,'position':position,'prompt':formatted,
                             'prompt_tokens':tokens,'gap_tokens':len(tokenizer.encode(gap_str,add_special_tokens=False))-
                               len(tokenizer.encode(formatted[:formatted.index('Passage:\n')],add_special_tokens=False)),
                             'question_tokens':len(tokenizer.encode(question,add_special_tokens=False))})
    if len(all_rows)!=48:raise RuntimeError('Missing 48 prompt conditions')
    for idx in range(16):
        base,pre,gap=(all_rows[idx],all_rows[16+idx],all_rows[32+idx])
        if abs(pre['prompt_tokens']-gap['prompt_tokens'])>6:raise RuntimeError('Prefix/gap length mismatch')
        if not (pre['prompt_tokens']>base['prompt_tokens']+700 and gap['prompt_tokens']>base['prompt_tokens']+700):
            raise RuntimeError('Context not sufficiently extended')
    receipt=sha_bytes(json.dumps([{'qid':r['qa_id'],'position':r['position'],
       'tokens':r['prompt_tokens'],'sha256':sha_bytes(r['prompt'].encode())}
       for r in all_rows],sort_keys=True,separators=(',',':')).encode())
    return all_rows,receipt,filler_tokens

def norm(text):
    # Strict full answer except punctuation, case and redundant whitespace.
    s=' '.join(text.strip().casefold().split())
    return re.sub(r'^[\s.,:;!"\']+|[\s.,:;!"\']+$','',s)

def score_one(row,output,hitcap):
    clean=norm(output)
    golds={norm(x) for x in row['answers']}
    abstained=(clean=='unknown')
    lenient=any(re.search(r'(?<!\w)'+re.escape(g)+r'(?!\w)',clean) for g in golds)
    return {'article':row['article_index'],'position':row['position'],
            'impossible':row['impossible'],'strict':int(abstained if row['impossible'] else clean in golds),
            'lenient':int(not row['impossible'] and lenient),
            'exact_unknown':int(abstained),'contains_unknown':int(bool(re.search(r'(?<!\w)unknown(?!\w)',clean))),
            'nonabstain_impossible':int(row['impossible'] and not abstained),
            'unexpected_unknown':int(not row['impossible'] and abstained),
            'hit_cap':int(hitcap),'empty':int(not clean),
            'tokens':row['prompt_tokens']}

def summarize(scored,position):
    rows=[r for r in scored if r['position']==position]
    a=[r for r in rows if not r['impossible']];u=[r for r in rows if r['impossible']]
    if len(rows)!=16 or len(a)!=8 or len(u)!=8:raise RuntimeError('Incomplete SQuAD subset')
    return {'answerable_exact':sum(r['strict'] for r in a),
            'answerable_lenient':sum(r['lenient'] for r in a),
            'unanswerable_UNKNOWN':sum(r['strict'] for r in u),
            'unsupported_nonabstention':sum(r['nonabstain_impossible'] for r in u),
            'unexpected_answerable_UNKNOWN':sum(r['unexpected_unknown'] for r in a),
            'token_caps':sum(r['hit_cap'] for r in rows),
            'empty':sum(r['empty'] for r in rows),
            'token_range':(min(r['tokens'] for r in rows),max(r['tokens'] for r in rows)),
            'n':len(rows)}

def render(stage,source_receipt,eligible,prompt_receipt,filler_n,scores,custody):
    lines=['# EXP-029 — Human-annotated SQuAD2 grounded generation and abstention',
      '', '**2026-10-14 | frozen-model CPU measured, research-only, n=8 answerable + 8 unanswerable paired articles.**',
      'Human-authored SQuAD 2.0 annotations provide gold; subset/filter/scoring is algorithmic,',
      'not independently human-adjudicated for this model prompt. SQuAD dev is publicly seen by model makers;',
      'this is an exploratory published-dev benchmark, not a hidden held-out test.',
      '', '## Exact sources and model identity','',
      'SQuAD 2.0 dev authors Rajpurkar et al., 2018; CC BY-SA 4.0,',
      'https://rajpurkar.github.io/SQuAD-explorer/ ; raw dataset excluded from Git.',
      f'Official dev-v2.0.json SHA256 `{SOURCE_SHA}`.',
      f'Preregistered protocol SHA256 `{PROTOCOL_SHA}`.',
      f'Chosen 8-topic 16-question gold/QA-ID source digest `{source_receipt}`;',
      f'deterministic eligible paired source paragraphs `{eligible}`.',
      f'Actual prompt/placement/tokenizer digest `{prompt_receipt}`.',
      f'Filler token target {FILL_TARGET}, observed {filler_n} (tokenizer-specific).',
      f"Model condition `{stage}`, pinned revision `{custody['revision']}`.",
      f"Model config SHA256 `{custody['model_config_sha256']}`; tokenizer SHA256 `{custody['tokenizer_sha256']}`.",
      f"Published native chat template used: **{custody['native_chat_template']}**.",
      f'Greedy decoding, max {MAX_NEW} new tokens, original model EOS; no answer-choice scoring.',
      '', '## Human-gold generated-answer and abstention results','',
      '| Placement | Context input tokens min–max | Strict factual exact /8 | Factual gold-span anywhere /8 | Strict UNKNOWN /8 unanswerable | Nonabstention /8 unanswerable | Answerable said UNKNOWN /8 | 32-token caps /16 |',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    for position in POSITIONS:
        x=scores[position]
        lines.append(f"| {position} | {x['token_range'][0]}–{x['token_range'][1]} | {x['answerable_exact']} | {x['answerable_lenient']} | {x['unanswerable_UNKNOWN']} | {x['unsupported_nonabstention']} | {x['unexpected_answerable_UNKNOWN']} | {x['token_caps']} |")
    lines+=['','## Positional contrasts','',
     'Identical model, source article, question, gold and filler. Short has no filler;',
     'prefix places the filler before the human passage; gap places it between passage',
     'and question. Prefix/gap total prompt lengths are within six model tokens per item.',
     'The following are **descriptive paired changes** over just eight articles.',
     '| Change | Strict factual difference /8 | Strict UNKNOWN difference /8 | All 16 token-cap difference |',
     '|---|---:|---:|---:|']
    for tag,a,b in (('prefix minus short','short','prefix'),('gap minus prefix','prefix','gap')):
        x=scores[a];y=scores[b]
        lines.append(f"| {tag} | {y['answerable_exact']-x['answerable_exact']:+d} | {y['unanswerable_UNKNOWN']-x['unanswerable_UNKNOWN']:+d} | {y['token_caps']-x['token_caps']:+d} |")
    lines+=['','## Necessary interpretation boundaries','',
       'A gold short answer is considered exact only when the *entire* generation',
       'matches a human span after minimal normalization. Lenient means the gold',
       'word is visible in longer output; it is **not** counted as strict compliance.',
       'An unanswerable response that does not exactly say UNKNOWN is nonabstention,',
       'not automatically a proven hallucination. Thirty-two-token cap rates',
       'are reported separately to prevent conflating truncation and semantics.',
       'One-word answer filter, article choice and answer-format instructions',
       'limit generalizability. Some human gold annotations can be ambiguous,',
       'and this is a known public dev split, not new blind evaluation.',
       'The experiment did not write any prompts, human gold, raw generations,',
       'text vectors, model weight files, NAS or production configurations.',
       'This does not establish K2 teacher ancestry, reliable long-context',
       'grounding or deployment readiness.',
       '', '## Reproduce under identical frozen model assets','',
       '1. Obtain official SQuAD2 dev-v2.0.json from the source URL above',
       '   under `/home/scott/git/.latent-squad2-dev-research/`; SHA must match.',
       '2. Use the isolated scottjoyner/lms research branch, Python venv and',
       '   pinned HF model checkpoints from EXP018/021.',
       '3. `python research/k2_human_grounded_qa.py --stage '+stage+' --evidence-dir /home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026 --out /home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026/EXPERIMENT-029-'+('K2' if stage=='mopd_249' else 'QWEN')+'-HUMAN-QA.md`',
       '4. The runner refuses existing output files and altered source/model versions.',
       'All score summaries are rendered directly as human-readable Markdown.', '']
    return '\n'.join(lines)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=STAGES,required=True)
    parser.add_argument('--evidence-dir',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--preflight-only',action='store_true')
    a=parser.parse_args()
    if a.out.exists():raise SystemExit('STOP report exists, no overwrite allowed')
    if sha(a.evidence_dir/'EXPERIMENT-029-PROTOCOL.md')!=PROTOCOL_SHA:
        raise RuntimeError('Preregistered protocol was modified')
    source,source_receipt,eligible=source_rows()
    sys.path.insert(0,str(a.evidence_dir))
    import run_exp021_free_response as prior
    if not a.preflight_only and ram()<38:raise RuntimeError('Resource admission <38 GiB')
    # Preflight uses only tokenizers and frozen local source; no large model load.
    from transformers import AutoTokenizer
    if a.stage=='mopd_249':
        import run_exp018_k2_intermediate_gates as e18
        root=e18.stage_path(a.stage)
    else:
        import run_exp014_exact_qwen08 as e14
        root=e14.QROOT
    tok=AutoTokenizer.from_pretrained(str(root),local_files_only=True,trust_remote_code=False)
    if a.stage=='qwen08_chat' and not tok.chat_template:raise RuntimeError('No official Qwen chat template')
    rows,prompt_sha,filler_n=generate_rows(tok,source,a.stage)
    print('EXP029_SOURCE_PREFLIGHT',a.stage,source_receipt,'paired_paragraphs',eligible,
          'prompt_digest',prompt_sha,'filler_n',filler_n,'range',
          [(pos,min(r['prompt_tokens'] for r in rows if r['position']==pos),
            max(r['prompt_tokens'] for r in rows if r['position']==pos)) for pos in POSITIONS],flush=True)
    if a.preflight_only:return
    model,tok,model_receipt=prior.load_model(a.stage)
    if a.stage=='qwen08_chat' and not model_receipt['native_chat_template']:
        raise RuntimeError('Wrong Qwen native chat provenance')
    if a.stage=='mopd_249' and model_receipt['native_chat_template']:
        raise RuntimeError('K2 cannot be labeled native chat')
    tok.padding_side='left'
    allscores=[]
    try:
        for offset in range(0,len(rows),BATCH):
            if ram()<22:raise RuntimeError('Resource gate <22GiB during inference')
            sub=rows[offset:offset+BATCH]
            tokens=tok([r['prompt'] for r in sub],padding=True,return_tensors='pt',
                       truncation=False,add_special_tokens=a.stage!='qwen08_chat')
            tokens.pop('token_type_ids',None)
            if tokens['input_ids'].shape[1]>1800:raise RuntimeError('Unexpected >1800-token context')
            with torch.inference_mode():
                outputs=model.generate(**tokens,do_sample=False,num_beams=1,
                    max_new_tokens=MAX_NEW,use_cache=True,
                    pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
            new=outputs[:,tokens['input_ids'].shape[1]:]
            for item,ids in zip(sub,new):
                ids=ids.tolist()
                stops=[i for i,t in enumerate(ids) if t==tok.eos_token_id]
                n=stops[0] if stops else len(ids)
                answer=tok.decode(ids[:n],skip_special_tokens=True)
                allscores.append(score_one(item,answer,not stops and n>=MAX_NEW))
            del tokens,outputs,new
            if offset%8==0:
                print('EXP029_PROGRESS',a.stage,offset+len(sub),'/',len(rows),
                      'elapsed_s',round(time.monotonic()-START,1),flush=True)
    finally:
        del model;gc.collect()
    if len(allscores)!=48:raise RuntimeError('Incomplete 48 direct generations')
    summary={pos:summarize(allscores,pos) for pos in POSITIONS}
    report=render(a.stage,source_receipt,eligible,prompt_sha,filler_n,summary,model_receipt)
    if not a.out.parent.is_dir():raise RuntimeError('Destination parent absent')
    scratch=a.out.with_name(a.out.name+'.tmp')
    if scratch.exists():raise RuntimeError('Scratch path already exists')
    scratch.write_text(report,encoding='utf-8');scratch.replace(a.out)
    print('EXP029_COMPLETED',a.stage,'report_sha256',sha(a.out),
          'summary',summary,'elapsed_s',round(time.monotonic()-START,1),flush=True)

if __name__=='__main__':main()
