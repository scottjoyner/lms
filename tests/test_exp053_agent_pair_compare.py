"""Synthetic-only paired source-hash and score validation tests."""
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
from exp053_agent_pair_compare import case_compare,summarize_pairs

def row(gold=-0.8,foil=-2.0):
    return {'case_id':'opaque-synthetic-case',
            'conditions':[{'arm':arm,'rotation':r,
                           'prompt_sha256':f'{r*3+j+1:064x}',
                           'gold_label':'A','foil_label':'B',
                           'gold_logp':float(gold),'foil_logp':float(foil)}
                          for r in (0,1) for j,arm in enumerate(('supported','withheld','decoy'))]}

def test_identical_sources_align_without_inference():
    q,m=case_compare(row(),row())
    assert q==m
    assert summarize_pairs([(q,m)])['D']['paired_qwen_minus_minicpm_mean']==0

def test_deterministic_paired_bootstrap():
    a,b=case_compare(row(),row(gold=-1.1))
    x=summarize_pairs([(a,b),(a,b)])
    assert x==summarize_pairs([(a,b),(a,b)])
    assert all(len(v['paired_bootstrap_95pct_descriptive'])==2 for v in x.values())

@pytest.mark.parametrize('mutation',[
 lambda r:r.update(case_id='different'),
 lambda r:r['conditions'][0].update(prompt_sha256='f'*64),
 lambda r:r['conditions'][0].update(gold_label='C'),
 lambda r:r['conditions'].pop(),
 lambda r:r['conditions'][0].update(gold_logp=float('nan')),
])
def test_bad_pair_fails_closed(mutation):
    q=row();m=row();mutation(m)
    with pytest.raises((ValueError,RuntimeError),match='HOLD'):
        case_compare(q,m)

def test_zero_case_fails_closed():
    with pytest.raises(ValueError,match='HOLD'):
        summarize_pairs([])
