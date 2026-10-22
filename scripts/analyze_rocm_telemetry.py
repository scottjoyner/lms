#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,statistics
from pathlib import Path

def walk_numbers(obj,prefix=''):
    if isinstance(obj,dict):
        for k,v in obj.items():
            yield from walk_numbers(v,f'{prefix}.{k}' if prefix else k)
    elif isinstance(obj,list):
        for i,v in enumerate(obj):
            yield from walk_numbers(v,f'{prefix}[{i}]')
    elif isinstance(obj,(int,float)) and not isinstance(obj,bool):
        yield prefix,float(obj)

def q(vals,p):
    if not vals:return None
    vals=sorted(vals);i=round((len(vals)-1)*p);return vals[i]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('input');ap.add_argument('--output',required=True);ap.add_argument('--tokens',type=float,default=0);args=ap.parse_args()
    rows=[json.loads(x) for x in Path(args.input).read_text().splitlines() if x.strip()]
    metrics={}
    rss=[r.get('rss_bytes') for r in rows if isinstance(r.get('rss_bytes'),(int,float))]
    for r in rows:
        for k,v in walk_numbers(r.get('amd_smi',{})):
            metrics.setdefault(k,[]).append(v)
    summary={'samples':len(rows),'rss_bytes':{'start':rss[0] if rss else None,'end':rss[-1] if rss else None,'delta':(rss[-1]-rss[0]) if len(rss)>=2 else None,'max':max(rss) if rss else None},'amd_smi':{}}
    for k,vals in metrics.items():
        summary['amd_smi'][k]={'mean':statistics.fmean(vals),'p50':q(vals,.5),'p95':q(vals,.95),'max':max(vals)}
    if args.tokens>0:
        power_candidates=[(k,v) for k,v in metrics.items() if 'power' in k.lower() or 'socket' in k.lower() and 'watt' in k.lower()]
        if power_candidates and len(rows)>1:
            k,vals=power_candidates[0]
            avg=statistics.fmean(vals);duration=max(0,len(rows)-1)
            summary['efficiency']={'power_metric':k,'avg_power_reported':avg,'approx_joules':avg*duration,'tokens_per_joule':args.tokens/(avg*duration) if avg*duration>0 else None}
    Path(args.output).write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print(json.dumps(summary,sort_keys=True))
if __name__=='__main__':main()
