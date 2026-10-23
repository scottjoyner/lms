#!/usr/bin/env python3
"""Fail-closed, read-only physical host-pressure admission for K2 EXP053 CPU batches.

No writes, downloads, privileged system inspection, service restarts or model loads.
An already occupied swap file is informational: pressure and swap *activity* matter.
"""
import argparse
from pathlib import Path
import time

GIB = 1024 ** 3
MAX_MEM_SOME_AVG10 = 1.5
MAX_MEM_FULL_AVG10 = 0.5
MAX_CPU_SOME_AVG10 = 15.
MIN_MEM_AVAILABLE_BYTES = 32 * GIB
MAX_SWAP_ACTIVITY_BYTES_PER_SECOND = 1024 * 1024


def parse_meminfo(contents):
    vals = {}
    for line in contents.splitlines():
        parts = line.replace(':', '').split()
        if len(parts) >= 3 and parts[-1] == 'kB':
            vals[parts[0]] = int(parts[1]) * 1024
    for key in ('MemAvailable', 'SwapTotal', 'SwapFree'):
        if key not in vals or vals[key] < 0:
            raise ValueError('HOLD EXP053 K2 missing meminfo '+key)
    return vals


def parse_psi(contents):
    result = {}
    for line in contents.splitlines():
        parts = line.split()
        if not parts or parts[0] not in ('some', 'full'):
            continue
        values = dict(x.split('=',1) for x in parts[1:] if '=' in x)
        result[parts[0]] = float(values['avg10'])
    if not result or any(v < 0 or not (v < float('inf')) for v in result.values()):
        raise ValueError('HOLD EXP053 K2 invalid pressure counters')
    return result


def parse_vmstat(contents):
    vals={}
    for line in contents.splitlines():
        k,*v=line.split()
        if k in ('pswpin','pswpout'):
            vals[k]=int(v[0])
    if set(vals)!= {'pswpin','pswpout'} or any(v<0 for v in vals.values()):
        raise ValueError('HOLD EXP053 K2 missing swap event counters')
    return vals


def probe():
    return (parse_meminfo(Path('/proc/meminfo').read_text()),
            parse_psi(Path('/proc/pressure/memory').read_text()),
            parse_psi(Path('/proc/pressure/cpu').read_text()),
            parse_vmstat(Path('/proc/vmstat').read_text()))


def inspect(before, after, elapsed):
    m, memory, cpu, swaps = after
    prior_swaps=before[3]
    if elapsed <= 0:
        raise ValueError('HOLD EXP053 K2 elapsed time invalid')
    swap_pages = sum(swaps[k]-prior_swaps[k] for k in ('pswpin','pswpout'))
    if swap_pages < 0:
        raise ValueError('HOLD EXP053 K2 swap counters regressed')
    # Linux pswpin/pswpout are page counters; use the actual host page size.
    import os
    page_size=os.sysconf('SC_PAGE_SIZE')
    swap_bps = swap_pages*page_size/elapsed
    checks={
       'mem_available':m['MemAvailable'] >= MIN_MEM_AVAILABLE_BYTES,
       'mem_some':memory['some'] <= MAX_MEM_SOME_AVG10,
       'mem_full':memory['full'] <= MAX_MEM_FULL_AVG10,
       'cpu_some':cpu['some'] <= MAX_CPU_SOME_AVG10,
       'swap_thrash':swap_bps <= MAX_SWAP_ACTIVITY_BYTES_PER_SECOND,
    }
    return checks, {
      'mem_available_GiB':round(m['MemAvailable']/GIB,1),
      'swap_occupied_GiB':round((m['SwapTotal']-m['SwapFree'])/GIB,1),
      'mem_some_avg10_pct':memory['some'],
      'mem_full_avg10_pct':memory['full'],
      'cpu_some_avg10_pct':cpu['some'],
      'swap_io_kib_s':round(swap_bps/1024,1)
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sample-seconds',type=float,default=4.)
    args=p.parse_args()
    if args.sample_seconds < 2 or args.sample_seconds > 10:
        raise SystemExit('EXP053_K2_RESOURCE_HOLD invalid observation window')
    before=probe()
    started=time.monotonic()
    time.sleep(args.sample_seconds)
    after=probe()
    checks, readout=inspect(before,after,time.monotonic()-started)
    print('EXP053_K2_HOST_PRESSURE',readout,flush=True)
    fails=[key for key,good in checks.items() if not good]
    if fails:
        raise SystemExit('EXP053_K2_RESOURCE_HOLD '+','.join(fails))
    print('EXP053_K2_RESOURCE_GO read_only_no_host_changes',flush=True)


if __name__ == '__main__':
    main()
