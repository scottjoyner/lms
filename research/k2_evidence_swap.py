#!/usr/bin/env python3
"""EXP034 source-swap causal suffix likelihoods; reports aggregate Markdown only."""
import argparse
import hashlib
import itertools
import json
import math
import os
import re
import sys
import time
from pathlib import Path
for k,v in {'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
            'CUDA_VISIBLE_DEVICES':'-1','OPENBLAS_NUM_THREADS':'1',
            'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}.items():os.environ[k]=v
import torch
import torch.nn.functional as F

PROTOCOL_SHA='d38072e36ef090606d6b765d077dd777af71cb4a65988bc92296ac7b7457c912'
SQUAD_RECEIPT='aad6e578544e2a3f100ab4b5c93d7ee6158fbfbe14bd426b8bffd9f9d1cde23b'
EXP029_K2_PROMPT='6513d65b456a0f52d230e386078d732a4b2800663c4d9b11c76b3cd569651c2a'
EXP029_QWEN_PROMPT='e27009abf39759a4b34a18665da2db87fc2a11aa9b53ecc9229ce37d4129f251'
STAGES=('mopd_249','qwen08_chat')
BATCH=2
MAX_PROMPT=1150
START=time.monotonic()
torch.set_num_threads(2)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def available():
    return next(int(l.split()[1]) for l in Path('/proc/meminfo').read_text().splitlines()
                if l.startswith('MemAvailable:'))/1024**2

def source_donors(rows,tok):
    original=[]
    for idx in range(8):
        matched=[r for r in rows if r['article_index']==idx]
        if len(matched)!=2 or matched[0]['context']!=matched[1]['context']:
            raise RuntimeError('Expected paired QA in one article')
        original.append(matched[0]['context'])
    sizes=[len(tok.encode(x,add_special_tokens=False)) for x in original]
    # DP-exhaustive 8! derangements; ties broken lexicographically.
    best=None
    for p in itertools.permutations(range(8)):
        if any(i==p[i] for i in range(8)):continue
        cost=sum(abs(sizes[i]-sizes[p[i]]) for i in range(8))
        if best is None or (cost,p)<best:best=(cost,p)
    cost,p=best
    assert len(p)==8 and len(set(p))==8 and all(i!=p[i] for i in range(8))
    differences=[abs(sizes[i]-sizes[p[i]]) for i in range(8)]
    digest=hashlib.sha256(json.dumps({'donors':p,'source_lengths':sizes},
        sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return p,sizes,differences,digest

def prep(evidence,prior,tokenizer,stage):
    if sha(evidence/'EXPERIMENT-034-PROTOCOL.md')!=PROTOCOL_SHA:
        raise RuntimeError('EXP034 preregistered protocol source changed')
    raw,receipt,eligible=prior.source_rows()
    if receipt!=SQUAD_RECEIPT or eligible!=90:raise RuntimeError('Human SQuAD source changed')
    from transformers import AutoTokenizer
    k2=AutoTokenizer.from_pretrained('/home/scott/git/.latent-k2-mopd_249-research',
                     local_files_only=True,trust_remote_code=False)
    donors,sizes,shifts,donor_sha=source_donors(raw,k2)
    tokens,prompt_sha,_=prior.generate_rows(tokenizer,raw,stage)
    expected=EXP029_K2_PROMPT if stage=='mopd_249' else EXP029_QWEN_PROMPT
    if prompt_sha!=expected:raise RuntimeError('Original native prompt drift')
    orig=[r for r in tokens if r['position']=='short']
    if len(orig)!=16:raise RuntimeError('Expected exactly 16 short QAs')
    pairs=[];leak_count=0
    for row in orig:
        own_idx=row['article_index']
        factual=next(q for q in raw if q['article_index']==own_idx and not q['impossible'])
        candidate=factual['answers'][0]
        donor_context=next(q for q in raw if q['article_index']==donors[own_idx])['context']
        if not row['impossible'] and re.search(r'(?<!\w)'+re.escape(candidate)+r'(?!\w)',
                                                donor_context,re.IGNORECASE):leak_count+=1
        old=f"Passage:\n{row['context']}\n"
        new=f"Passage:\n{donor_context}\n"
        if row['prompt'].count(old)!=1:raise RuntimeError('Original passage boundary ambiguous')
        swapped=row['prompt'].replace(old,new,1)
        if swapped==row['prompt'] or row['question'] not in swapped:
            raise RuntimeError('Donor passage failed to change precisely')
        for kind,prompt in (('original',row['prompt']),('donor',swapped)):
            pairs.append({'article':own_idx,'impossible':row['impossible'],
                          'source':kind,'prompt':prompt,'candidate':candidate,
                          'question_id':row['qa_id']})
    if len(pairs)!=32 or len({(r['question_id'],r['source']) for r in pairs})!=32:
        raise RuntimeError('Incomplete source swap pairs')
    suffixes=[]
    # Prefix parity must hold for both full candidate strings and BOTH models.
    add_special=stage!='qwen08_chat'
    for row in pairs:
        prefix=tokenizer.encode(row['prompt'],add_special_tokens=add_special)
        if len(prefix)>MAX_PROMPT:raise RuntimeError('Overlong candidate prompt')
        for label,value in (('answer',row['candidate']),('unknown','UNKNOWN')):
            full=tokenizer.encode(row['prompt']+' '+value,add_special_tokens=add_special)
            if full[:len(prefix)]!=prefix:raise RuntimeError('Tokenizer boundary merge; no first-token proxy')
            suffix=full[len(prefix):]
            if not 1<=len(suffix)<=8:raise RuntimeError('Missing or excessive complete suffix')
            suffixes.append({'article':row['article'],'impossible':row['impossible'],
                             'source':row['source'],'choice':label,'prefix_len':len(prefix),
                             'full':full,'suffix':suffix})
    receipt=hashlib.sha256(json.dumps([{'id':r['question_id'],'source':r['source'],
        'prompt_sha':hashlib.sha256(r['prompt'].encode()).hexdigest()} for r in pairs],
        sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return suffixes,receipt,donor_sha,shifts,leak_count

def batch_logps(model,head,samples,stage,pad_id):
    maxlen=max(len(s['full']) for s in samples)
    # Top-level Qwen3.5 config lacks pad_token_id. The published tokenizer
    # is the authoritative padding source for the attention-masked batch.
    if not isinstance(pad_id,int) or pad_id<0 or pad_id>=int(head.shape[0]):
        raise RuntimeError('Invalid tokenizer-native pad ID')
    pad=pad_id
    ids=torch.full((len(samples),maxlen),pad,dtype=torch.long)
    attn=torch.zeros_like(ids)
    for i,s in enumerate(samples):
        ids[i,:len(s['full'])]=torch.tensor(s['full'])
        attn[i,:len(s['full'])]=1
    with torch.inference_mode():
        if stage=='qwen08_chat':
            hs=model.model.language_model(input_ids=ids,attention_mask=attn,
                              use_cache=False,return_dict=True).last_hidden_state
        else:
            hs=model.model(input_ids=ids,attention_mask=attn,
                              use_cache=False).last_hidden_state
        rows=[];positions=[];targets=[];sizes=[]
        for idx,s in enumerate(samples):
            n=len(s['suffix']);sizes.append(n)
            rows.extend([idx]*n)
            positions.extend(range(s['prefix_len']-1,s['prefix_len']-1+n))
            targets.extend(s['suffix'])
        vectors=hs[torch.tensor(rows),torch.tensor(positions)]
        logits=(vectors@head.T).float()
        lp=F.log_softmax(logits,dim=-1)
        result=lp[torch.arange(len(targets)),torch.tensor(targets)]
        if not torch.isfinite(result).all():raise RuntimeError('Nonfinite conditional likelihood')
        offset=0;out=[]
        for n in sizes:
            out.append(float(result[offset:offset+n].sum().item()));offset+=n
    return out

def summarize(scored):
    by={}
    for row in scored:
        key=(row['article'],row['impossible'],row['source'],row['choice'])
        if key in by:raise RuntimeError('Double scored candidate')
        by[key]=row['lp']
    if len(by)!=64:raise RuntimeError('Incomplete 64 real suffix scores')
    result={}
    for impossible in (False,True):
        diffs=[];base=[];swap=[];topics=[]
        for article in range(8):
            def margin(src):return by[(article,impossible,src,'answer')]-by[(article,impossible,src,'unknown')]
            x,y=margin('original'),margin('donor')
            diffs.append(x-y);base.append(x);swap.append(y)
            topics.append((int(x>0),int(y>0)))
        ds=sorted(diffs)
        result['unanswerable' if impossible else 'answerable']={
          'n':len(diffs),'positive_source_gain':sum(v>0 for v in diffs),
          'negative_source_gain':sum(v<0 for v in diffs),
          'mean_source_gain':sum(diffs)/8,'median_source_gain':(ds[3]+ds[4])/2,
          'min_source_gain':min(diffs),'max_source_gain':max(diffs),
          'original_positive_margin':sum(v>0 for v in base),
          'donor_positive_margin':sum(v>0 for v in swap),
          'mean_original_margin':sum(base)/8,'mean_donor_margin':sum(swap)/8}
    return result

def render(stage,modelreceipt,prompt_sha,donorsha,length_shifts,answer_leaks,stats,elapsed):
    lines=['# EXP-034 — Human-annotated evidence-swap conditional-likelihood witness',
       '',f'**2026-10-14 | Frozen model `{stage}`; actual complete candidate-sequence CPU scoring.**',
       'Only eight public SQuAD2 source-article groups; not independent new gold.',
       '', '## Exact custody','',
       f'Protocol SHA256 `{PROTOCOL_SHA}`; fixed SQuAD2 source selection `{SQUAD_RECEIPT}`.',
       f'Original SQuAD2 short prompt receipt `{EXP029_K2_PROMPT if stage=="mopd_249" else EXP029_QWEN_PROMPT}`.',
       f'Paired article source-derangement SHA256 `{donorsha}`.',
       f'Actual 32 prompt/position receipt `{prompt_sha}`.',
       f"Model official revision `{modelreceipt['revision']}`.",
       f"Model config SHA256 `{modelreceipt['model_config_sha256']}`; tokenizer SHA256 `{modelreceipt['tokenizer_sha256']}`.",
       f"Published native chat template used: **{stage=='qwen08_chat'}**.",
       'Original passage or a different original SQuAD2 article passage, same question',
       'and frozen answer candidate/UNKNOWN strings in each paired condition.',
       f'Tokenization-only donor-length shifts: min {min(length_shifts)}, median {sorted(length_shifts)[3:5]}, max {max(length_shifts)} tokens.',
       f'Human-gold candidate appears incidentally in unrelated donor: **{answer_leaks}/8 answerable cases**.',
       'No model generation, answer truncation, sampling or changed learned weights.',
       '', '## Full candidate log-likelihood controls','',
       'Scores are *sum of full suffix token log probabilities*, with full native',
       'vocabulary softmax, not first-token-only or free-form answer validity.',
       'Article answer word is the positive candidate even for paired unanswerable',
       'questions, where it is merely an **unsupported foil** versus UNKNOWN.',
       '', '| Question class | Original margin >0 /8 | Donor margin >0 /8 | Original-minus-donor margin >0 /8 | Mean ΔlogP | Median ΔlogP | Min..max Δ |',
       '|---|---:|---:|---:|---:|---:|---:|']
    for kind in ('answerable','unanswerable'):
        r=stats[kind]
        lines.append(f"| {kind} | {r['original_positive_margin']} | {r['donor_positive_margin']} | {r['positive_source_gain']} | {r['mean_source_gain']:+.5f} | {r['median_source_gain']:+.5f} | {r['min_source_gain']:+.4f}..{r['max_source_gain']:+.4f} |")
    lines+=['','The sign of `(original - donor)` indicates whether the original',
       'passage increases the relative probability of its article answer word',
       'over UNKNOWN, **not** whether the model would produce a correct answer.',
       'For unanswerable questions, a positive margin for the article-associated',
       'foil can be a failure of evidence discrimination.',
       '', '## Interpretation boundaries','',
       'Source swapping changes topic and passage semantics and only coarsely matches',
       'lengths. Public SQuAD2 development text might occur in training data.',
       'The foil is a word from the original article, not a human-written plausible',
       'wrong answer. Raw sums favor shorter candidate tokenizations; within-question',
       'paired source deltas cancel this fixed candidate-length penalty, but',
       'original-vs-donor decisions should not be treated as calibrated abstention.',
       'Model formats differ for K2 plain completion versus Qwen native chat.',
       'No teacher genealogy, hidden-neuron identity, evidence entailment,',
       'NO-GO overturn or model admission is established from this pilot.',
       'This study is research-only, no training, NAS, router, graph or production change.',
       '',f'Execution elapsed {elapsed:.2f} seconds; all 64 candidate strings completed.',
       '', '## Reproduction','', '```bash',
       'python research/k2_evidence_swap.py --stage '+stage+' --evidence-dir /home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026 --out /home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026/EXPERIMENT-034-'+('K2' if stage=='mopd_249' else 'QWEN')+'-EVIDENCE-SWAP.md',
       '```','', 'Original SQuAD2 human annotations © Rajpurkar et al., 2018, CC BY-SA 4.0:',
       'https://rajpurkar.github.io/SQuAD-explorer/','']
    return '\n'.join(lines)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=STAGES,required=True)
    parser.add_argument('--evidence-dir',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args()
    if args.out.exists():raise SystemExit('STOP existing report')
    sys.path.insert(0,str(args.evidence_dir))
    import k2_human_grounded_qa as prior
    import run_exp021_free_response as backend
    from transformers import AutoTokenizer
    if args.stage=='mopd_249':
        import run_exp018_k2_intermediate_gates as e18
        modelroot=e18.stage_path(args.stage)
    else:
        import run_exp014_exact_qwen08 as e14
        modelroot=e14.QROOT
    tok=AutoTokenizer.from_pretrained(str(modelroot),local_files_only=True,trust_remote_code=False)
    seq,receipt,donor,shifts,leaks=prep(args.evidence_dir,prior,tok,args.stage)
    print('EXP034_PREFLIGHT',args.stage,'n_suffixes',len(seq),'src',receipt,
          'donor',donor,'length_shifts',shifts,'answer_leaks',leaks,flush=True)
    if args.preflight_only:return
    if available()<36:raise RuntimeError('Host memory below 36 GiB initial gate')
    model,tok,modelreceipt=backend.load_model(args.stage)
    expected=('7abae15437c2e7f1b2cd986be562344f204a17f3' if args.stage=='mopd_249'
              else '2fc06364715b967f1860aea9cf38778875588b17')
    if modelreceipt['revision']!=expected:raise RuntimeError('Model custody mismatch')
    head=model.lm_head.weight.detach()
    scored=[]
    try:
        for start in range(0,len(seq),BATCH):
            if available()<22:raise RuntimeError('In-run memory below 22 GiB')
            group=seq[start:start+BATCH]
            values=batch_logps(model,head,group,args.stage,tok.pad_token_id)
            for row,value in zip(group,values):
                scored.append({'article':row['article'],'impossible':row['impossible'],
                               'source':row['source'],'choice':row['choice'],'lp':value})
            if start%16==0:
                print('EXP034_PROGRESS',args.stage,start+len(group),'/',len(seq),
                      'elapsed_s',round(time.monotonic()-START,1),flush=True)
        output=render(args.stage,modelreceipt,receipt,donor,shifts,leaks,
                      summarize(scored),time.monotonic()-START)
        if not args.out.parent.is_dir():raise RuntimeError('Report dir absent')
        tmp=args.out.with_name(args.out.name+'.tmp')
        if tmp.exists():raise RuntimeError('Scratch report exists')
        tmp.write_text(output,encoding='utf8');tmp.replace(args.out)
        print('EXP034_COMPLETE',args.stage,'report_SHA',sha(args.out),flush=True)
    finally:
        del model

if __name__=='__main__':main()
