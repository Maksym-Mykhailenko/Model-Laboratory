#!/usr/bin/env python3
"""Materialize compact Stage-6 records into chat SFT JSONL for a later Stage-7 trainer."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from model_lab.interpreter_training import load_jsonl, materialize_sft_record

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument('input',type=Path); p.add_argument('output',type=Path); args=p.parse_args()
    items=load_jsonl(args.input); args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',encoding='utf-8',newline='\n') as f:
        for item in items: f.write(json.dumps(materialize_sft_record(item),ensure_ascii=False,separators=(',',':'))+'\n')
    print(f'Wrote {len(items)} SFT records to {args.output}')
    return 0
if __name__=='__main__': raise SystemExit(main())
