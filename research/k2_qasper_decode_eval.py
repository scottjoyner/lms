#!/usr/bin/env python3
"""EXP047 frozen 16-v-64 K2 evidence and deterministic answer-span experiment."""
import os
for k,v in {'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','CUDA_VISIBLE_DEVICES':'-1',
'TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'2'}.items():os.environ[k]=v
import argparse,hashlib,time,gc
from pathlib import Path
import torch
from transformers import StoppingCriteriaList
from k2_interpolation_pilot import (MID,RL,TOK_SHA,IDX_SHA,CONFIG_SHA,preflight,check_shard_layout,sha,available_gib)
from k2_first_newline_stopping import StopAtGeneratedNewline
from k2_qasper_decode_fixture import cohort,prompt,receipt,score,aggregate,MODES,CAPS,KINDS
ROOT=Path('/home/scott/git/wt-knowledge-k2-latex-20261014/50-Research/llm-research/latent-world-geometry-2026')
PROTOCOL_SHA='f83d358b61a653afd2d42bf04b252d906a69bbf8186a8725a8890ca347af4deb'
COHORT_SHA='0f9c4fad7ba905a3b1283911ea50b77f4470b710183509425e24bd14328f69c5'
torch.set_num_threads(2)
def frozen():
 if sha(ROOT/'EXPERIMENT-047-PROTOCOL-2026-10-15.md')!=PROTOCOL_SHA:raise RuntimeError('HOLD protocol SHA drift')
 for i,root in enumerate((MID,RL)):
  if not root.is_dir() or root.is_symlink():raise RuntimeError('HOLD missing pinned model')
  for filename,expected in (('model.safetensors.index.json',IDX_SHA[i]),('config.json',CONFIG_SHA[i]),('tokenizer.json',TOK_SHA)):
   if sha(root/filename)!=expected:raise RuntimeError('HOLD model artifact SHA drift')
 rows=cohort()
 if len(rows)!=16 or receipt(rows)!=COHORT_SHA:raise RuntimeError('HOLD QASPER cohort hash')
 return rows

def generate(model,tok,row,limit):
 if available_gib()<22:raise RuntimeError('HOLD available memory below 22GiB')
 ins=tok(prompt(row),return_tensors='pt',truncation=False,add_special_tokens=True)
 ins.pop('token_type_ids',None)
 n=int(ins['input_ids'].shape[1])
 if not 20<n<750:raise RuntimeError('HOLD input length')
 pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
 stop=StopAtGeneratedNewline(tok,n)
 with torch.inference_mode():
  out=model.generate(**ins,do_sample=False,num_beams=1,max_new_tokens=limit,use_cache=True,
   stopping_criteria=StoppingCriteriaList([stop]),pad_token_id=pad,
   eos_token_id=tok.eos_token_id,return_dict_in_generate=True,output_scores=True)
 seq=out.sequences
 if seq.ndim!=2 or seq.shape[0]!=1:raise RuntimeError('HOLD sequence topology')
 ids=seq[0,n:].tolist()
 if not 1<=len(ids)<=limit or len(out.scores)!=len(ids):raise RuntimeError('HOLD decode scores/length')
 if any(not torch.isfinite(x).all().item() for x in out.scores):raise RuntimeError('HOLD nonfinite logits')
 eos=tok.eos_token_id
 eoses=set(eos) if isinstance(eos,(tuple,list)) else {eos}
 pos=next((i for i,v in enumerate(ids) if v in eoses),len(ids))
 raw=tok.decode(ids[:pos],skip_special_tokens=True)
 newline='\n' in raw
 if newline!=bool(stop.hit):raise RuntimeError('HOLD inconsistent newline stopper')
 visible=raw.split('\n',1)[0] if newline else raw
 native_eos=pos<len(ids)
 cap=pos>=limit and not newline and not native_eos
 result=score(row,visible,hit_newline=newline,cap=cap,eos=native_eos)
 return result,n,ids

def report(results,paired,details,seconds,prompt_tokens):
 lines=['# EXP-047 — Longer K2 generation versus deterministic answer-span extraction',
  '', '**2026-10-15 | 64 actual paired K2 CPU generations; draft research only.**',
  '', '## Custody and design', '',
  '- Preregistered immutable protocol SHA256: '+PROTOCOL_SHA,
  '- Untouched last eight official QASPER 0.3 eligible research paper topics;',
  '  all topics disjoint from EXP045 and EXP046 (24 previous papers).',
  '- Full 16-prompt/gold cohort SHA256: '+COHORT_SHA,
  '- Same Mid2 BF16 learned weights, default RoPE versus native RL YaRN.',
  '- Each pair repeated with 16 versus 64 newly generated tokens, greedy,',
  '  first newly decoded newline stopping, no source/label/prompt changes.',
  '- Original QASPER human full-paper unanswerable labels are evaluated',
  '  against only one excerpt, not against complete source document.',
  '- PRIMARY strict normalized full answer unchanged from EXP045.',
  '- SECONDARY fixed prefix-stripped first-sentence span parser is neither',
  '  a gold substring lookup nor an independently human-adjudicated answer.',
  '', '## Complete full-output / span diagnostic', '',
  '| RoPE | Token cap | Human category | N | Strict exact | Span exact | Strict F1 | Span F1 | UNKNOWN | Newline stop | EOS | Cap hit | Empty |',
  '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for mode in MODES:
  for limit in CAPS:
   for kind in KINDS:
    d=results[(mode,limit)][kind]
    lines.append(f"| {mode} | {limit} | {kind} | {d['n']} | {d['strict']} | {d['extracted_exact']} | {d['f1']:.6f} | {d['extracted_f1']:.6f} | {d['unknown']} | {d['newline']} | {d['eos']} | {d['cap']} | {d['empty']} |")
 lines+=['','## Paired 16-token to 64-token changes','',
  '| RoPE | Human category | N | Strict gained | Strict lost | Span gained | Span lost | Short token-prefix of long |',
  '|---|---|---:|---:|---:|---:|---:|---:|']
 for mode in MODES:
  for kind in KINDS:
   d=paired[(mode,kind)]
   lines.append(f"| {mode} | {kind} | 8 | {d['strict_gained']} | {d['strict_lost']} | {d['span_gained']} | {d['span_lost']} | {d['prefix_equal']} |")
 lines+=['','## Limits and release boundaries','',
  '- Original strict scores and diagnostic parser scores cannot be pooled.',
  '- Eight public research-paper clusters, human FULL-PAPER no-answer labels',
  '  against partial excerpts, possible pretraining exposure, unreviewed paraphrases.',
  '- Longer outputs may change format without improving evidence comprehension.',
  f'- Prompt token counts min/max: {min(prompt_tokens)}/{max(prompt_tokens)}.',
  f'- Actual native inference wall duration: {seconds:.2f}s.',
  '- No source text, article ID, raw completion, per-item label, model weights,',
  '  activations or logits persisted. No train/merge/serving/routing, graph/NAS,',
  '  network/credential change, deployment or arXiv submission. PRs draft.','']
 return '\n'.join(lines)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');ap.add_argument('--out',type=Path);args=ap.parse_args()
 rows=frozen();print('EXP047_PREFLIGHT_OK',len(rows),COHORT_SHA[:12],flush=True)
 if not args.execute:print('EXP047_DRY_RUN_NO_MODEL',flush=True);return
 if args.out is None or args.out.exists() or args.out.parent.resolve()!=ROOT.resolve():
  raise RuntimeError('HOLD report path exists or not isolated')
 idx,_,_=preflight()
 if check_shard_layout(*idx)!=1078285824:raise RuntimeError('HOLD parameter inventory')
 from transformers import AutoTokenizer,AutoConfig,AutoModelForCausalLM
 tok=AutoTokenizer.from_pretrained(str(MID),local_files_only=True,trust_remote_code=False)
 prompts=[len(tok.encode(prompt(row),add_special_tokens=True)) for row in rows]
 if len(prompts)!=16 or not all(20<n<750 for n in prompts):raise RuntimeError('HOLD prompt length')
 print('EXP047_TOKENIZED',min(prompts),max(prompts),flush=True)
 cfg=AutoConfig.from_pretrained(str(RL),trust_remote_code=True,local_files_only=True)
 if cfg.rope_parameters['rope_type']!='yarn':raise RuntimeError('HOLD YaRN config')
 if available_gib()<36:raise RuntimeError('HOLD model load available memory below 36GiB')
 model=AutoModelForCausalLM.from_pretrained(str(MID),trust_remote_code=True,
     local_files_only=True,dtype=torch.bfloat16,use_safetensors=True,low_cpu_mem_usage=True)
 model.eval().to('cpu').requires_grad_(False)
 original=model.model.rotary_emb
 alternative=type(original)(cfg,device=torch.device('cpu'))
 if original.rope_type!='default' or alternative.rope_type!='yarn':raise RuntimeError('HOLD rotary modules')
 results={}; ephemeral={};timer=time.monotonic()
 try:
  for mode,rot in (('default',original),('yarn',alternative)):
   model.model.rotary_emb=rot
   for limit in CAPS:
    scores=[];sequences=[]
    for j,row in enumerate(rows):
     v,n,ids=generate(model,tok,row,limit);scores.append(v);sequences.append(ids)
     if (j+1)%4==0:print('EXP047_PROGRESS',mode,limit,j+1,'/',len(rows),'s',round(time.monotonic()-timer,1),flush=True)
    if len(scores)!=16:raise RuntimeError('HOLD incomplete condition')
    results[(mode,limit)]=aggregate(rows,scores)
    ephemeral[(mode,limit)]=(scores,sequences)
    print('EXP047_ARM_COMPLETE',mode,limit,[(k,results[(mode,limit)][k]['strict'],results[(mode,limit)][k]['extracted_exact']) for k in KINDS],flush=True)
  if len(results)!=4:raise RuntimeError('HOLD incomplete 64 generations')
  paired={}
  for mode in MODES:
   small=ephemeral[(mode,16)];large=ephemeral[(mode,64)]
   for kind in KINDS:
    grp=[(a,b,x,y) for row,a,b,x,y in zip(rows,small[0],large[0],small[1],large[1]) if row['kind']==kind]
    if len(grp)!=8:raise RuntimeError('HOLD paired category')
    paired[(mode,kind)]={
      'strict_gained':sum(not a['strict'] and b['strict'] for a,b,_,_ in grp),
      'strict_lost':sum(a['strict'] and not b['strict'] for a,b,_,_ in grp),
      'span_gained':sum(not a['extracted_exact'] and b['extracted_exact'] for a,b,_,_ in grp),
      'span_lost':sum(a['extracted_exact'] and not b['extracted_exact'] for a,b,_,_ in grp),
      'prefix_equal':sum(y[:len(x)]==x for _,_,x,y in grp)}
  out=report(results,paired,None,time.monotonic()-timer,prompts)
  args.out.open('x',encoding='utf8').write(out)
  print('EXP047_COMPLETE',hashlib.sha256(out.encode()).hexdigest(),flush=True)
 finally:
  model.model.rotary_emb=original
  del model,alternative
  gc.collect()
if __name__=='__main__':main()
