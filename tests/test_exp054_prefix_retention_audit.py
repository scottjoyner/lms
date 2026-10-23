"""Synthetic-only private source retention contracts."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"research"))
import exp054_prefix_retention_audit as tool


def sample():
    lookup={f"case{i:03d}":{"article":"S"*(2000+i*2)} for i in range(64)}
    source=[{"case_id":f"case{i:03d}","decoy_case_id":f"case{(i+1)%64:03d}",
             "supported_prefix_codepoints":1000,
             "decoy_prefix_codepoints":1000} for i in range(64)]
    return lookup,source


def test_stable_aggregate_without_source_text():
    lookup,source=sample()
    v=tool.analyze(lookup,source)
    assert set(v)=={"supported","decoy"}
    assert v["supported"]["article_count"]==64
    assert v["supported"]["mean_prefix_codepoints"]==1000.
    assert v["supported"]["below_one_quarter_article"]==0
    assert v==tool.analyze(lookup,source)


@pytest.mark.parametrize("mutate",[
 lambda d:d[0].pop("case000"),
 lambda d:d[1].pop(),
 lambda d:d[1][0].update(supported_prefix_codepoints=200),
 lambda d:d[1][0].update(supported_prefix_codepoints=1700),
 lambda d:d[1][0].update(supported_prefix_codepoints=500),
 lambda d:d[1][0].update(decoy_case_id="case999"),
 lambda d:d[0]["case000"].update(article=""),
 lambda d:d[0]["case000"].update(article="x"*300),
])
def test_fail_closed_source_drift(mutate):
    data=list(sample())
    mutate(data)
    with pytest.raises((ValueError,KeyError),match="HOLD|case999"):
        tool.analyze(*data)


def test_never_load_model():
    code=(Path(__file__).resolve().parents[1]/"research"/"exp054_prefix_retention_audit.py").read_text()
    assert "from_pretrained" not in code
    assert "model(" not in code
