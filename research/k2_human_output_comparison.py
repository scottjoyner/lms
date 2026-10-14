#!/usr/bin/env python3
"""EXP030 audited Markdown-only comparison, no raw gold, logits or output text."""
import argparse
import hashlib
import re
from pathlib import Path

REPORTS={
 'strict32':('EXPERIMENT-030-STRICT32.md','4bf704a6b64ef79f0ca5b73cfe2e55b73c8a39990ad5328c8a90367614ba175e'),
 'budget64':('EXPERIMENT-030-BUDGET64.md','ac206d810c20668e22c790f687c803be55ef868e02ab846bd576bec12b40b490'),
}
BASE=('EXPERIMENT-029-K2-HUMAN-QA.md','0ab5ac2d744362969b97618b32ce9ad536ae32bda820a4bfac79410cc7c17c71')
PROTOCOL='c046e294c766385e9500e47cf971d4b8d7421683456e43bc29e2ad367f868727'
ROW=re.compile(r'^\| (short|gap) \| (\d+–\d+) \| (\d+) \| (\d+) \| (\d+) \| (\d+) \| (\d+) \| (\d+) \| (\d+) \|$')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def inputs(directory):
    if sha(directory/BASE[0])!=BASE[1]:raise RuntimeError('EXP029 measured parent changed')
    if sha(directory/'EXPERIMENT-030-PROTOCOL.md')!=PROTOCOL:
        raise RuntimeError('EXP030 protocol changed')
    result={}
    for arm,(name,receipt) in REPORTS.items():
        p=directory/name
        if sha(p)!=receipt:raise RuntimeError('EXP030 measured arm changed: '+arm)
        text=p.read_text()
        if PROTOCOL not in text or BASE[1] not in text:
            raise RuntimeError('Unproven prior/protocol custody')
        rows={}
        for line in text.splitlines():
            m=ROW.fullmatch(line)
            if not m:continue
            place,span,*rest=m.groups()
            if place in rows:raise RuntimeError('Duplicate arm-place result')
            values=list(map(int,rest))
            if len(values)!=7 or any(x>8 for x in values[:-1]) or values[-1]>16:
                raise RuntimeError('Invalid audited scores')
            rows[place]=dict(zip(('exact','lenient','first_gold','unknown',
                   'first_unknown','nonabstain','caps'),values))
            rows[place]['tokens']=span
            if values[3]+values[5]!=8:
                raise RuntimeError('Contradictory UNKNOWN accounting')
        if set(rows)!=set(('short','gap')):raise RuntimeError('Arm incomplete')
        result[arm]=rows
    return result

def render(measure):
    lines=['# EXP-030 — K2 human-annotated output contract and budget comparison',
       '', '**2026-10-14 | complete source-locked, frozen-model, two-factor-isolated research.**',
       'Two *separate* interventions on eight SQuAD2 answerable + eight',
       'unanswerable human-annotated questions, each in short and long-gap',
       'conditions. Raw model generations and human passage text are not stored.',
       '', '## Source and exact audit trail','',
       'Original public SQuAD 2.0 dev: Rajpurkar et al. (2018), CC BY-SA 4.0.',
       'https://rajpurkar.github.io/SQuAD-explorer/',
       f'Pre-inference EXP030 protocol SHA256 `{PROTOCOL}`.',
       f'Original 32-token K2 EXP029 Markdown SHA256 `{BASE[1]}`.',
       '| New arm | Independent measured Markdown SHA256 |', '|---|---|']
    for arm,(_,digest) in REPORTS.items():lines.append(f'| {arm} | `{digest}` |')
    lines+=['','## Human-gold results by controlled condition','',
      '| Arm | Placement | Strict gold /8 | Gold present /8 | First gold word /8 | Strict UNKNOWN /8 | First UNKNOWN /8 | No abstention /8 | Generation caps /16 |',
      '|---|---|---:|---:|---:|---:|---:|---:|']
    base={'short':(0,7,0,9),'gap':(0,6,0,14)}
    for place,(strict,lenient,unknown,caps) in base.items():
        lines.append(f'| baseline32 (EXP029) | {place} | {strict} | {lenient} | not measured | {unknown} | not measured | 8 | {caps} |')
    for arm,rows in measure.items():
        for place in ('short','gap'):
            x=rows[place]
            lines.append(f"| {arm} | {place} | {x['exact']} | {x['lenient']} | {x['first_gold']} | {x['unknown']} | {x['first_unknown']} | {x['nonabstain']} | {x['caps']} |")
    lines+=['','## Diagnostic conclusions, and what is NOT established','',
       '**Strict-instruction only, 32 tokens:** all 32 outputs reached the',
       'token cap; 0 strict gold, 0 exact UNKNOWN, while the first word',
       'matched the gold in 5/8 answerable cases per position. An external',
       'output-boundary rule might salvage first words, but would not',
       'enable trusted abstention. Never count first-word match as whole',
       'response correctness.',
       '', '**Budget-only, 64 tokens:** original inputs were exactly unchanged;',
       'strict gold and UNKNOWN remain zero under both contexts. Gold',
       'word appears in 8/8 short and 7/8 gap, but can occur in verbose',
       'noncompliant content; output caps persisted 9/16 short and 13/16 gap.',
       'Doubling max tokens did not correct output control or abstention.',
       '', 'The original EXP029 short/gap cases are repeated eight-source',
       'article comparisons, not 32 independent judgments. All source gold',
       'labels are pre-existing public SQuAD2 annotations; human gold',
       'word presence is a lenient proxy only. The study does not establish',
       'K2 teacher identity, general grounding, fail-safe UNKNOWN behavior,',
       'or production model admission. K2 grounded QA stays NO-GO.',
       '', '## Reproduce and verify','',
       'The frozen LMS `research/k2_human_output_contract.py` runner must',
       'produce the two upstream reports with the audited official K2',
       'mOPD checkpoint and SQuAD2 dev source. The independent comparison',
       'then runs:', '', '```bash',
       'python research/k2_human_output_comparison.py --evidence-dir /home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026 --out /home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026/EXPERIMENT-030-COMPARISON.md',
       '```','']
    return '\n'.join(lines)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise SystemExit('STOP existing comparison output')
    result=render(inputs(args.evidence_dir))
    if not args.out.parent.is_dir():raise RuntimeError('Destination parent absent')
    tmp=args.out.with_name(args.out.name+'.tmp')
    if tmp.exists():raise RuntimeError('Scratch collision')
    tmp.write_text(result,encoding='utf8');tmp.replace(args.out)
    print('EXP030_COMPARISON_PASS',sha(args.out),'lines',len(result.splitlines()))
if __name__=='__main__':main()
