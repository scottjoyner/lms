#!/usr/bin/env python3
"""EXP029: deterministic SHA-pinned comparative review of two measured Markdown reports.

No new inference, no access to answer texts, no standalone machine JSON.
"""
import argparse
import hashlib
import re
from pathlib import Path

SOURCE_SHA='80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8'
GOLD_SELECTION='aad6e578544e2a3f100ab4b5c93d7ee6158fbfbe14bd426b8bffd9f9d1cde23b'
PROTOCOL='547ec8dcbe91e821604864160a7c3f340fa43bff7b88be0dbf36ddf72323ad68'
SHA={
 'K2':('EXPERIMENT-029-K2-HUMAN-QA.md','0ab5ac2d744362969b97618b32ce9ad536ae32bda820a4bfac79410cc7c17c71'),
 'Qwen (native chat)':('EXPERIMENT-029-QWEN-HUMAN-QA.md','a166f1727620bc2cc162f07c388584a97201bf2064e233cb92c87c2d3d60cd85'),
}
POSITIONS=('short','prefix','gap')
ROW=re.compile(r'^\| (short|prefix|gap) \| (\d+–\d+) \| (\d+) \| (\d+) \| (\d+) \| (\d+) \| (\d+) \| (\d+) \|$')

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def inputs(directory):
    measurements={}
    for model,(name,expected) in SHA.items():
        file=directory/name
        if digest(file)!=expected:raise RuntimeError('Measured source SHA mismatch '+name)
        doc=file.read_text()
        for checksum in (SOURCE_SHA,GOLD_SELECTION,PROTOCOL):
            if checksum not in doc:raise RuntimeError('Dataset/protocol receipt disagreement')
        expected_chat=(model!='K2')
        if f'Published native chat template used: **{expected_chat}**' not in doc:
            raise RuntimeError('Model prompt provenance mismatch')
        rows={}
        for line in doc.splitlines():
            match=ROW.fullmatch(line)
            if not match:continue
            p,rng,*n=match.groups()
            if p in rows:raise RuntimeError('Duplicate measured position')
            vals=tuple(map(int,n))
            if len(vals)!=6 or any(v>8 for v in vals[:5]) or vals[5]>16:
                raise RuntimeError('Reported count out of bounds')
            rows[p]={'input_range':rng,'exact':vals[0], 'contains':vals[1],
                     'unknown':vals[2],'nonabstain':vals[3],
                     'unexpected_unknown':vals[4],'caps':vals[5]}
        if set(rows)!=set(POSITIONS):raise RuntimeError('Incomplete source positions')
        for v in rows.values():
            if v['unknown']+v['nonabstain']!=8:raise RuntimeError('Abstention count mismatch')
        measurements[model]=rows
    return measurements

def render(measured):
    lines=['# EXP-029 — Human-gold grounded generation: K2 versus native-chat Qwen',
       '', '**2026-10-14 | reproducible secondary comparison of two COMPLETE model passes.**',
       'This compares **different model architectures and different published',
       'prompt formats**, so it cannot isolate architecture, learned weights,',
       'instruction tuning, tokenizer or chat-template effects.',
       '', '## Evidence custody','',
       f'Official SQuAD 2.0 human-authored dev set SHA256 `{SOURCE_SHA}`; CC BY-SA 4.0.',
       'Source: https://rajpurkar.github.io/SQuAD-explorer/',
       f'Deterministic selected eight-source-article receipt `{GOLD_SELECTION}`.',
       f'Pre-inference preregistration with transparent resource amendment `{PROTOCOL}`.',
       'Both 48-generation passes are complete, greedy with 32-token budget.',
       'No raw question, gold or model output text is read by this comparison.',
       '','| Model | Measured Markdown report SHA256 |', '|---|---|']
    for model,(name,sha) in SHA.items():
        lines.append(f'| {model} — `{name}` | `{sha}` |')
    lines+=['','## Complete human-gold outcome comparison','',
        'Each condition has eight answerable and eight human-annotated',
        'unanswerable questions from the *same* eight Wikipedia-source articles.',
        '| Model | Filler placement | Input tokens | Exact answerable /8 | Gold somewhere /8 | Exact UNKNOWN /8 | Nonabstention /8 | 32-token caps /16 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for model,positions in measured.items():
        for position in POSITIONS:
            a=positions[position]
            lines.append(f"| {model} | {position} | {a['input_range']} | {a['exact']} | {a['contains']} | {a['unknown']} | {a['nonabstain']} | {a['caps']} |")
    lines+=['','## Across repeated placements (NOT independent article replicates)','',
        '| Model | Strict factual /24 | Gold identifiable /24 | Strict UNKNOWN /24 | Token caps /48 |',
        '|---|---:|---:|---:|---:|']
    for model,results in measured.items():
        totals={metric:sum(r[metric] for r in results.values())
                for metric in ('exact','contains','unknown','caps')}
        lines.append(f"| {model} | {totals['exact']} | {totals['contains']} | {totals['unknown']} | {totals['caps']} |")
    lines+=['','## What these results do and do not mean','',
        '**K2** often included the human-gold short answer inside its output',
        '(7/8, 8/8, 6/8 across the three placements), yet never exactly',
        'returned the answer or UNKNOWN. It reached the 32-token cap on',
        '34/48 outputs. Presence of a correct span in long text is not usable',
        'strict grounded QA, nor a proof the whole generated explanation was correct.',
        '', '**Native-chat Qwen** exactly returned gold on 5/8 answerable',
        'questions in all three placements. Strict UNKNOWN was 2/8 short',
        'and 6/8 under both long placements, with only one token-cap hit',
        'in 48 generations. It still did not abstain on 2/8 unanswerable',
        'cases even with long filler and once answered UNKNOWN to a',
        'gold-answerable question in each long condition.',
        '', 'Surprisingly, **adding neutral filler improved Qwen abstention**',
        'irrespective of whether filler preceded or separated the evidence.',
        'This is compatible with a prompt-length/formatting dependency,',
        'not proof of better long-range memory or safer general refusal.',
        '', 'All 24 rows/condition share just eight source articles: these totals',
        'are dependent repeated observations and cannot be used as',
        'confidence intervals or comprehensive model rankings. The public',
        'SQuAD dev split may appear in training data, and the word-answer',
        'mechanical filter is selection biased. Gold questions are human',
        'annotated but this custom prompt was not newly human adjudicated.',
        'K2 plain prompting is NOT comparable to Qwen official native chat',
        'for isolated architecture causality. These results do not identify',
        'a K2 training teacher, establish model-layer ancestry or authorize',
        'deployment, graph effects, router promotion or NAS mutation.',
        '', '## Exact reproducibility','',
        '1. Recreate the two source reports using the pinned code and original',
        '   SQuAD2 dev file per `EXPERIMENT-029-PROTOCOL.md`.',
        '2. Run `python research/k2_human_qa_compare.py --evidence-dir /home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026 --out /home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026/EXPERIMENT-029-COMPARISON.md`.',
        '3. Script fails closed if source hash, human-gold selection,',
        '   measured count inventory or published-chat provenance changes.', '']
    return '\n'.join(lines)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise SystemExit('STOP: target report already exists')
    comparisons=inputs(args.evidence_dir)
    text=render(comparisons)
    if not args.out.parent.is_dir():raise RuntimeError('Missing output parent')
    scratch=args.out.with_name(args.out.name+'.tmp')
    if scratch.exists():raise RuntimeError('Scratch collision')
    scratch.write_text(text,encoding='utf8');scratch.replace(args.out)
    print('EXP029_COMPARISON_COMPLETE',digest(args.out),len(text.splitlines()))

if __name__=='__main__':main()
