#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, csv, json
from pathlib import Path

COLS = [
    ('method','Method'),('N','N'),('Info_Gain','Info Gain ↑'),('Query_Efficiency','Query Eff. ↑'),
    ('Revision_Accuracy','Revision Acc. ↑'),('Final_Strict','Final Strict ↑'),('Unsafe_Rate','Unsafe ↓'),
    ('Trace_Consistency','Trace ↑'),('Agent_OSRS','Agent OSRS ↑')
]

def load(p): return json.loads(Path(p).read_text(encoding='utf-8'))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--summaries', nargs='+', required=True)
    ap.add_argument('--out_md', required=True)
    ap.add_argument('--out_csv', required=True)
    args=ap.parse_args()
    rows=[]
    for p in args.summaries:
        if not Path(p).exists():
            print(f'[WARN] missing summary skipped: {p}')
            continue
        s=load(p)
        rows.append({
            'method': s.get('method','unknown'), 'N': s.get('N',0),
            'Info_Gain': s.get('Info_Gain',0), 'Query_Efficiency': s.get('Query_Efficiency',0),
            'Revision_Accuracy': s.get('Revision_Accuracy',0), 'Final_Strict': s.get('Final_Strict',0),
            'Unsafe_Rate': s.get('Unsafe_Rate',0), 'Trace_Consistency': s.get('Trace_Consistency',0),
            'Agent_OSRS': s.get('Agent_OSRS',0),
        })
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv,'w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f, fieldnames=[k for k,_ in COLS]); w.writeheader(); w.writerows(rows)
    header='| ' + ' | '.join(h for _,h in COLS) + ' |'
    sep='|'+'|'.join(['---']*len(COLS))+'|'
    lines=[header, sep]
    for r in rows:
        lines.append('| ' + ' | '.join(str(r[k]) for k,_ in COLS) + ' |')
    Path(args.out_md).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'[OK] wrote {args.out_md}')
    print(f'[OK] wrote {args.out_csv}')
if __name__=='__main__': main()
