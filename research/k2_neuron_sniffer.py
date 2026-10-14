#!/usr/bin/env python3
"""EXP032: frozen read-only MLP gated-neuron study, source-locked Markdown."""
import argparse, gc, hashlib, itertools, json, os, re, sys, time
from pathlib import Path
for k,v in {'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','CUDA_VISIBLE_DEVICES':'-1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}.items():os.environ[k]=v
import numpy as np
import torch
PROTOCOL='db83b66726ab0d75ce3370a512e278768f90db0fd81009daa01a2966cedb30d9'
SOURCE_SHA='aad6e578544e2a3f100ab4b5c93d7ee6158fbfbe14bd426b8bffd9f9d1cde23b'
ORIGINAL_SHA={'mid2_47684':'6513d65b456a0f52d230e386078d732a4b2800663c4d9b11c76b3cd569651c2a','rl_merged':'6513d65b456a0f52d230e386078d732a4b2800663c4d9b11c76b3cd569651c2a','qwen08_chat':'e27009abf39759a4b34a18665da2db87fc2a11aa9b53ecc9229ce37d4129f251'}
STAGES=('mid2_47684','rl_merged','qwen08_chat')
LAYERS={'mid2_47684':(0,7,14,27),'rl_merged':(0,7,14,27),'qwen08_chat':(0,6,12,23)}
REVISIONS={'mid2_47684':'', 'rl_merged':'', 'qwen08_chat':'2fc06364715b967f1860aea9cf38778875588b17'}
START=time.monotonic(); torch.set_num_threads(2)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def gib():return next(int(x.split()[1]) for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))/1024**2

def cka(x,y):
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    if x.shape[0]!=y.shape[0] or x.shape[0]!=16:raise ValueError('unmatched 16 samples')
    k=x@x.T;l=y@y.T
    k-=k.mean(axis=0,keepdims=True);k-=k.mean(axis=1,keepdims=True)
    l-=l.mean(axis=0,keepdims=True);l-=l.mean(axis=1,keepdims=True)
    d=np.linalg.norm(k)*np.linalg.norm(l)
    if d<1e-22:return None
    return float(np.sum(k*l)/d)

def paired_max(a):
    # 16 examples in sorted article order: [answerable, impossible] each.
    diff=np.asarray(a)[0::2]-np.asarray(a)[1::2]
    if diff.shape[0]!=8:raise ValueError('paired article source layout broken')
    denom=np.sqrt(np.mean(diff.astype('float64')**2,axis=0))+1e-9
    normed=diff.astype('float64')/denom
    observed=float(np.max(np.abs(normed.mean(axis=0))))
    counts=0
    for bits in itertools.product((-1.,1.),repeat=8):
        null=float(np.max(np.abs((normed*np.array(bits)[:,None]).mean(axis=0))))
        if null>=observed-1e-12:counts+=1
    return observed,counts/256

def neuron_stats(x):
    v=np.asarray(x,dtype='float64')
    if v.ndim!=2 or v.shape[0]!=16 or not np.isfinite(v).all():raise ValueError('Bad neuron activation shape')
    m=np.mean(np.abs(v),axis=0)
    return {'width':int(v.shape[1]),'abs_mean':float(m.mean()),'zero_frac':float(np.mean(v==0)),
            'top32_mass':float(np.sort(m)[-32:].sum()/max(m.sum(),1e-15)),
            'pair_score':paired_max(v)[0],'null_fraction':paired_max(v)[1]}

def same_coordinate(x,y):
    x=np.asarray(x,dtype='float64'); y=np.asarray(y,dtype='float64')
    if x.shape!=y.shape or x.shape[0]!=16:raise ValueError('K2 native neuron dimension mismatch')
    scale=np.linalg.norm(x,axis=1)*np.linalg.norm(y,axis=1)
    cos=np.sum(x*y,axis=1)/np.maximum(scale,1e-30)
    ma=np.mean(abs(x),axis=0);mb=np.mean(abs(y),axis=0)
    ix=set(np.argsort(ma)[-32:].tolist());iy=set(np.argsort(mb)[-32:].tolist())
    return float(cos.mean()),len(ix&iy)

def source_prompts(prior,tok,stage):
    cases,receipt,count=prior.source_rows()
    if receipt!=SOURCE_SHA or count!=90:raise RuntimeError('Human source changed')
    rows,sh,filler=prior.generate_rows(tok,cases,stage)
    if sh!=ORIGINAL_SHA[stage] or filler!=768:raise RuntimeError('Input source/tokenizer mismatch '+stage)
    original=[r for r in rows if r['position']=='short']
    if len(original)!=16 or sum(r['impossible'] for r in original)!=8:raise RuntimeError('Expected paired 16')
    if [r['article_index'] for r in original]!=[i for i in range(8) for _ in range(2)]:raise RuntimeError('source order mismatch')
    return original

def capture(model,tok,rows,stage):
    ids=LAYERS[stage]
    module_by_id={}
    for name,mod in model.named_modules():
        m=re.search(r'\.layers\.(\d+)\.mlp\.down_proj$',name)
        if m and int(m.group(1)) in ids:
            if int(m.group(1)) in module_by_id:raise RuntimeError('Duplicate decoder module')
            module_by_id[int(m.group(1))]=mod
    if set(module_by_id)!=set(ids):
        raise RuntimeError(f'Wrong discovered down_proj modules: expected {ids} found {list(module_by_id)}')
    values={i:[] for i in ids}; one={}
    def hook(idx):
        def before(mod,args):
            if idx in one:raise RuntimeError('MLP down projection invoked more than once per sample')
            inp=args[0]
            if inp.ndim!=3 or inp.shape[0]!=1:raise RuntimeError('Incorrect activation input shape')
            one[idx]=inp[0,-1,:].detach().to('cpu',dtype=torch.float32).numpy().copy()
        return before
    handles=[module_by_id[i].register_forward_pre_hook(hook(i)) for i in ids]
    try:
        for j,row in enumerate(rows):
            if gib()<22:raise RuntimeError('Resource stop <22 GiB')
            one.clear()
            x=tok(row['prompt'],return_tensors='pt',truncation=False,
                  add_special_tokens=stage!='qwen08_chat')
            x.pop('token_type_ids',None)
            if x['input_ids'].shape[1]>1300:raise RuntimeError('Exceeded frozen short context limit')
            with torch.inference_mode():out=model(**x,use_cache=False)
            if set(one)!=set(ids):raise RuntimeError('One or more MLP hooks failed')
            for i in ids:values[i].append(one[i]);
            del x,out
            if j%4==3:print('EXP032',stage,'source',j+1,'/16',round(time.monotonic()-START,1),flush=True)
    finally:
        for h in handles:h.remove()
    return {i:np.stack(v,axis=0) for i,v in values.items()}

def report(all_values,custody):
    lines=['# EXP-032 — Frozen MLP neuron activation sniffer across K2 and Qwen architectures',
      '', '**2026-10-14 | source-pinned real BF16 CPU forwards, research only.**',
      'This is an observational MLP activation study, not causal neuron or teacher identification.',
      '', '## Exact custody and structural boundaries','',
      f'EXP032 preregistration SHA256 `{PROTOCOL}`; SQuAD selected source SHA256 `{SOURCE_SHA}`.',
      'All 16 exact EXP029 short prompts from 8 paired original SQuAD2 article topics.',
      'Four prespecified relative-depth blocks per model; native gated-MLP',
      '`down_proj` **input** activations at last input token per prompt.',
      'No raw per-question activations/neuron IDs, answers, text or vectors saved.',
      '| Frozen stage | Official revision | Model-config SHA256 | Original prompt receipt |',
      '|---|---|---|---|']
    for stage in STAGES:
        receipt=custody[stage]
        lines.append(f"| {stage} | `{receipt['revision']}` | `{receipt['model_config_sha256']}` | `{ORIGINAL_SHA[stage]}` |")
    lines+=['','## Actual MLP intermediate activation statistics (not neuron role assignments)','',
      '| Stage | Decoder layer | Gated width | Mean abs activation | Exactly-zero fraction | Top-32 mean-abs mass fraction | Paired answerability max statistic | Full max-null fraction (256 flips) |',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    for stage in STAGES:
        for layer in LAYERS[stage]:
            s=neuron_stats(all_values[stage][layer])
            lines.append(f"| {stage} | {layer} | {s['width']} | {s['abs_mean']:.6g} | {s['zero_frac']:.5f} | {s['top32_mass']:.5f} | {s['pair_score']:.5f} | {s['null_fraction']:.5f} |")
    lines+=['','## Within-K2 same-topology neuron coordinate similarity','',
            '| K2 matched native layer | Mean matched-prompt gated-neuron cosine | Top-32 same-coordinate overlap /32 |',
            '|---:|---:|---:|']
    for idx in LAYERS['mid2_47684']:
        cos,ov=same_coordinate(all_values['mid2_47684'][idx],all_values['rl_merged'][idx])
        lines.append(f'| {idx} | {cos:.6f} | {ov} |')
    lines+=['','## K2-to-Qwen matched-example CKA of gated MLP activations','',
      'Cross-model neuron IDs are **NOT** aligned; sample Gram matrices are',
      '16×16, with per-model tokenizer and prompt formatting. Exact example',
      'source order is matched, not token indices or hidden dimensions.',
      '| Approximate depth | K2 RL layer | Qwen layer | Matched-sample CKA | 64 article-pair shuffled-null mean |',
      '|---|---:|---:|---:|---:|']
    rng=np.random.default_rng(20261014)
    for depth,(ki,qi) in enumerate(zip(LAYERS['rl_merged'],LAYERS['qwen08_chat'])):
        x=all_values['rl_merged'][ki];y=all_values['qwen08_chat'][qi]
        original=cka(x,y);shuffled=[]
        for _ in range(64):
            perm=rng.permutation(8)
            rows=[2*int(a)+int(b) for a in perm for b in (0,1)]
            shuffled.append(cka(x,y[rows]))
        if original is None or any(v is None for v in shuffled):raise RuntimeError('Degenerate activation Gram matrices')
        lines.append(f'| {depth+1}/4 | {ki} | {qi} | {original:.6f} | {float(np.mean(shuffled)):.6f} |')
    lines+=['', '## Limits and next decisive study','',
       'Within-K2 matching neuron coordinates is meaningful only as a',
       'same-topology checkpoint diagnostic; high cosine does not imply a',
       'shared human-interpretable concept. Across K2/Qwen, widths and',
       'linear-attention structure differ: neither single-neuron IDs nor',
       'raw activation amplitudes should be treated as equivalent.',
       'The paired-answerability nulls are exact 2^8 article-level sign flips,',
       'with a max-over-neurons statistic recomputed each time. Eight pairs',
       'are *far too few* for a neuron discovery claim or independent p-value.',
       'Qwen native chat versus K2 plain formatting is confounded.',
       'The output describes source-conditioned activation associations only,',
       'not causal steering, representational homology, teacher provenance,',
       'or credible generalization to unseen data. Follow-up should use',
       'source-held-out articles, neuron ablation with negative controls,',
       'matched instruction conditions, and an explicit abstention gate.',
       'No training, graph, credentials, routing, NAS or model promotion.',
       '', '## Replay','',
       'Use published checkpoint/SQuAD source hashes with the frozen LMS script',
       '`research/k2_neuron_sniffer.py` and protocol `EXPERIMENT-032-PROTOCOL.md`.',
       'The runner refuses overwritten reports and any incomplete model stage.','']
    return '\n'.join(lines)

def main():
    c=argparse.ArgumentParser(description=__doc__)
    c.add_argument('--evidence-dir',required=True,type=Path)
    c.add_argument('--out',required=True,type=Path)
    x=c.parse_args()
    if x.out.exists():raise SystemExit('STOP existing Markdown report')
    if sha(x.evidence_dir/'EXPERIMENT-032-PROTOCOL.md')!=PROTOCOL:raise RuntimeError('Prospective protocol changed')
    sys.path.insert(0,str(x.evidence_dir))
    import k2_human_grounded_qa as prior
    import run_exp021_free_response as backend
    from transformers import AutoTokenizer
    import run_exp018_k2_intermediate_gates as exp18
    import run_exp014_exact_qwen08 as exp14
    activations={};custody={}
    for stage in STAGES:
        if gib()<36:raise RuntimeError('Resource HOLD before stage '+stage)
        root=exp14.QROOT if stage=='qwen08_chat' else exp18.stage_path(stage)
        tokenizer=AutoTokenizer.from_pretrained(str(root),local_files_only=True,trust_remote_code=False)
        rows=source_prompts(prior,tokenizer,stage)
        model,tok,receipt=backend.load_model(stage)
        try:
            if stage=='qwen08_chat' and receipt['revision']!=REVISIONS[stage]:raise RuntimeError('Qwen revision changed')
            values=capture(model,tok,rows,stage)
            custody[stage]=receipt
            activations[stage]=values
        finally:
            del model; gc.collect()
        print('EXP032_STAGE_COMPLETE',stage,'secs',round(time.monotonic()-START,1),flush=True)
    text=report(activations,custody)
    if not x.out.parent.exists():raise RuntimeError('Output parent missing')
    tmp=x.out.with_name(x.out.name+'.tmp')
    if tmp.exists():raise RuntimeError('Scratch path exists')
    tmp.write_text(text,encoding='utf8');tmp.replace(x.out)
    print('EXP032_FULL_COMPLETE',sha(x.out),'secs',round(time.monotonic()-START,1))
if __name__=='__main__':main()
