"""Synthetic-only resource gate checks, independent of real host pressure."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
import exp053_k2_resource_gate as gate

MIB=1024**2


def healthy():
    mem={'MemAvailable':40*gate.GIB,'SwapTotal':8*gate.GIB,'SwapFree':0}
    return mem, {'some':0.,'full':0.}, {'some':0.,'full':0.}, {'pswpin':100,'pswpout':200}


def test_full_but_idle_swap_is_safe():
    same=healthy()
    checks,stats=gate.inspect(same,same,4)
    assert all(checks.values())
    assert stats['swap_occupied_GiB']==8
    assert stats['swap_io_kib_s']==0


@pytest.mark.parametrize('mutate,fails',[
    (lambda x:x[0].update(MemAvailable=30*gate.GIB),'mem_available'),
    (lambda x:x[1].update(some=2.0),'mem_some'),
    (lambda x:x[1].update(full=1.0),'mem_full'),
    (lambda x:x[2].update(some=20.0),'cpu_some'),
    (lambda x:x[3].update(pswpout=1000000),'swap_thrash')
])
def test_reject_pressure(mutate,fails):
    before=healthy()
    after=healthy()
    mutate(after)
    checks,_=gate.inspect(before,after,4)
    assert not checks[fails]


def test_missing_or_invalid_meminfo_holds():
    with pytest.raises(ValueError,match='HOLD'):
        gate.parse_meminfo('MemAvailable: 100000 kB\nSwapTotal: 0 kB\n')
    with pytest.raises(ValueError,match='HOLD'):
        gate.parse_vmstat('pswpin 0\n')


def test_invalid_pressure_holds():
    with pytest.raises(ValueError,match='HOLD'):
        gate.parse_psi('some avg10=NaN avg60=0 avg300=0 total=100')


def test_swap_counter_regression_holds():
    before=healthy();after=healthy()
    after[3]['pswpin']=-1
    with pytest.raises(ValueError,match='HOLD'):
        gate.inspect(before,after,4)


def test_entrypoint_checks_resources_per_batch():
    root=Path(__file__).resolve().parents[1]
    batch=(root/'research/exp053_k2_remaining_batches.sh').read_text()
    assert batch.count('python3 -B research/exp053_k2_resource_gate.py')>=2
    assert batch.index('exp053_k2_resource_gate.py') < batch.index('for start in')
