#!/usr/bin/env python3
"""EXP034 audit two BF16 Markdown source-swap witnesses; no raw data read."""
import argparse
import hashlib
import re
from pathlib import Path

REPORTS={
 'K2 plain':('EXPERIMENT-034-K2-EVIDENCE-SWAP.md','eea7b12d7de64a74057721006028b2cbef2e70c113d3dd10cfde3610d1cf2661'),
 'Qwen native chat':('EXPERIMENT-034-QWEN-EVIDENCE-SWAP.md','403105ee08da20907de6850574a340544397bb98b5da55f830c13d00685f014d'),
}
PROTOCOL='d38072e36ef090606d6b765d077dd777af71cb4a65988bc92296ac7b7457c912'
GOLD='aad6e578544e2a3f100ab4b5c93d7ee6158fbfbe14bd426b8bffd9f9d1cde23b'
DONOR='bd5004731519d1b5b6c994885e5f519bb7ed5209616e329937812395a9759380'
ROW=re.compile(r'^\| (answerable|unanswerable) \| (\d+) \| (\d+) \| (\d+) \| ([+-]\d+\.\d+) \| ([+-]\d+\.\d+) \| ([+-]\d+\.\d+)\.\.([+-]\d+\.\d+) \|$')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def inputs(root):
    if sha(root/'EXPERIMENT-034-PROTOCOL.md')!=PROTOCOL:
        raise RuntimeError('Preregistered protocol changed')
    out={}
    for model,(name,hashvalue) in REPORTS.items():
        if sha(root/name)!=hashvalue:raise RuntimeError('Measured source changed: '+name)
        doc=(root/name).read_text()
        for receipt in (PROTOCOL,GOLD,DONOR):
            if receipt not in doc:raise RuntimeError('Unproven science custody '+name)
        expected_chat=(model=='Qwen native chat')
        if f'Published native chat template used: **{expected_chat}**.' not in doc:
            raise RuntimeError('Model format mismatch')
        metrics={}
        for line in doc.splitlines():
            m=ROW.fullmatch(line)
            if not m:continue
            cohort,a,b,sign,mean,median,lo,hi=m.groups()
            if cohort in metrics:raise RuntimeError('Duplicate cohort')
            a,b,sign=int(a),int(b),int(sign)
            if any(v>8 for v in (a,b,sign)) or not 0<=a+b+sign:
                raise RuntimeError('Out-of-range count')
            values=dict(original=a,donor=b,positive=sign,mean=float(mean),
                        median=float(median),min=float(lo),max=float(hi))
            if values['min']>values['median'] or values['median']>values['max']:
                raise RuntimeError('Invalid signed margin')
            metrics[cohort]=values
        if set(metrics)!={'answerable','unanswerable'}:
            raise RuntimeError('Missing cohort result')
        out[model]=metrics
    return out

def render(measure):
    lines=['# EXP-034 — Source-word presence versus question-specific grounding, K2 and Qwen',
      '', '**2026-10-14 | complete real-model, human-annotated public-dev source-swap comparison.**',
      'The parent studies compare original source paragraphs with different',
      'article paragraphs under fixed Q&A questions and candidate suffixes.',
      'This reconciler reads ONLY SHA-pinned aggregate Markdown; no logits,',
      'raw SQuAD source text or generated answers are available to it.',
      '', '## Exact experimental custody', '',
      'SQuAD2 human-annotated dev: Rajpurkar et al. 2018, CC BY-SA 4.0.',
      'https://rajpurkar.github.io/SQuAD-explorer/',
      f'Preregistered source-swap protocol SHA256 `{PROTOCOL}`.',
      f'Original human-source selection SHA256 `{GOLD}`.',
      f'Gold-blind K2-token length-nearest donor derangement SHA256 `{DONOR}`.',
      'One original SQuAD2 article-answer-word candidate per question',
      'versus UNKNOWN; 8 answerable + 8 unanswerable article pairs per model.',
      '| Source evidence | Exact source SHA256 |', '|---|---|']
    for model,(name,h) in REPORTS.items():
        lines.append(f'| {model} — `{name}` | `{h}` |')
    lines+=['', '## Actual complete-string conditional likelihood results','',
      '| Model format | Human question class | Original favors article word /8 | Donor favors word /8 | Source raises article-word margin /8 | Mean within-model ΔlogP |',
      '|---|---|---:|---:|---:|---:|']
    for model,classes in measure.items():
        for group in ('answerable','unanswerable'):
            x=classes[group]
            lines.append(f"| {model} | {group} | {x['original']} | {x['donor']} | {x['positive']} | {x['mean']:+.5f} |")
    lines+=['','## What changes the decision','',
      '**Positive passage dependence is not necessarily entailment.** Both',
      'models increase their article-word margin when the original article',
      'replaces a donor, for all eight answerable questions **and** all',
      'eight unanswerable paired questions. The latter intentionally tests',
      'an article word that answers another question, not this one.',
      '', 'In the *original* unanswerable passages, K2 prefers the unsupported',
      'article-word foil to UNKNOWN in **5/8** cases; native-chat Qwen in',
      '**1/8**. This is consistent with the previously observed Qwen advantage',
      'on abstention, but is not proof of a trusted entailment verifier.',
      '', '**Do not compare the absolute mean logP differences numerically',
      'across K2 and Qwen**: their vocabularies, candidate subword lengths,',
      'architectures and prompt formats differ. The within-model signed',
      'source effect and human-negative preference counts are interpretable',
      'as exploratory diagnostics, not calibrated binary classifiers.',
      'Donor source swapping also changes topic and has imperfect K2-token',
      'length matching (3–64 tokens). All eight public-dev source articles',
      'have been exposed in earlier studies, not independently held out.',
      'No inference about teacher genealogy or full neuron homology.',
      'K2 grounded QA remains NO-GO for production.',
      '', '## Reproduction','',
      'Recompute two upstream Markdown reports with pinned models and official',
      'SQuAD2 source via `research/k2_evidence_swap.py` and verify their hashes.',
      'Then run:', '```bash',
      'python research/k2_evidence_swap_compare.py --evidence-dir /home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026 --out /home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026/EXPERIMENT-034-COMPARISON.md',
      '```', '', 'Research-only; no raw passages, NAS, graph, deployment, training,',
      'model registry or production router mutation.', '']
    return '\n'.join(lines)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise SystemExit('STOP existing report output')
    report=render(inputs(a.evidence_dir))
    tmp=a.out.with_name(a.out.name+'.tmp')
    if tmp.exists():raise RuntimeError('Scratch collision')
    tmp.write_text(report,encoding='utf-8');tmp.replace(a.out)
    print('EXP034_COMPARISON_COMPLETE',sha(a.out),len(report.splitlines()))

if __name__=='__main__':main()
