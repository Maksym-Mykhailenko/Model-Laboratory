#!/usr/bin/env python3
"""Build the deterministic Stage-6 interpreter corpus."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from model_lab.interpreter_training import DEFAULT_SEED, write_corpus

def main()->int:
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=ROOT/'training'/'interpreter_corpus_v1.5')
    p.add_argument('--seed',type=int,default=DEFAULT_SEED)
    p.add_argument('--workers',type=int,default=None)
    args=p.parse_args()
    manifest=write_corpus(args.output,seed=args.seed,workers=args.workers)
    print(json.dumps(manifest,indent=2))
    return 0
if __name__=='__main__': raise SystemExit(main())
