"""Synthetic-only three-model paired contrasts, without corpus or model inputs."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"research"))
import exp053_agent_three_model_compare as three


def samples(count=2):
    q,m,k=three.LABELS
    return [{q:{"D":3.+i, "N":-1., "S":4.+i},
             m:{"D":2.+i, "N":-1., "S":3.+i},
             k:{"D":1.+i, "N":-1., "S":2.+i}}
            for i in range(count)]


def test_three_model_deterministic_synthetic():
    a=three.analyze(samples())
    assert a == three.analyze(samples())
    assert a["models"][three.LABELS[0]]["D"]["mean"] == 3.5
    assert a["paired_differences"][three.LABELS[0]+" minus "+three.LABELS[2]]["D"]["mean"] == 2.
    assert a["paired_differences"][three.LABELS[0]+" minus "+three.LABELS[2]]["D"]["paired_article_bootstrap_95pct_descriptive"] == [2.,2.]


def test_bootstrap_covers_negative_and_null():
    a=three.analyze(samples())
    assert a["paired_differences"][three.LABELS[0]+" minus "+three.LABELS[1]]["N"]["mean"] == 0.


@pytest.mark.parametrize("mutation",[
    lambda x:x[0][three.LABELS[0]].update(D=float("nan")),
    lambda x:x[0][three.LABELS[2]].update(S=float("inf")),
    lambda x:x[1][three.LABELS[1]].pop("D"),
])
def test_three_model_tamper_holds(mutation):
    rows=samples()
    mutation(rows)
    with pytest.raises((ValueError, KeyError),match="HOLD|D"):
        three.analyze(rows)


def test_empty_cohort_fails_closed():
    with pytest.raises(ValueError,match="HOLD"):
        three.analyze([])
