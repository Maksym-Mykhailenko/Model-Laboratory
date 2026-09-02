#!/usr/bin/env python3
"""Validate Stage-6 corpus hashes, split isolation, leakage, review binding and compiler grounding."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from model_lab import __version__
from model_lab.canonical import canonical_json_sha256
from model_lab.interpreter_training import (
    CORPUS_SCHEMA, CORPUS_SCHEMA_VERSION, load_jsonl, validate_corpus_examples,
    validate_corpus_examples_parallel, validate_review_queue
)


def main()->int:
    p=argparse.ArgumentParser()
    p.add_argument('--corpus',type=Path,default=ROOT/'training'/'interpreter_corpus_v1.5')
    p.add_argument('--deep',action='store_true')
    p.add_argument('--workers',type=int,default=None,help='Worker count for deep compiler replay.')
    p.add_argument('--output',type=Path,default=None,help='Optional path for the machine-readable validation report.')
    args=p.parse_args()
    manifest_path=args.corpus/'manifest.json'
    manifest_raw=manifest_path.read_bytes()
    manifest=json.loads(manifest_raw.decode('utf-8'))
    if not isinstance(manifest,dict) or manifest.get('schema')!=CORPUS_SCHEMA or manifest.get('schema_version')!=CORPUS_SCHEMA_VERSION:
        raise SystemExit('Unsupported or malformed corpus manifest')
    declared_manifest_sha=manifest.get('manifest_sha256')
    without_sha={k:v for k,v in manifest.items() if k!='manifest_sha256'}
    if declared_manifest_sha!=canonical_json_sha256(without_sha):
        raise SystemExit('manifest.json canonical SHA-256 does not verify')

    train=load_jsonl(args.corpus/'train.jsonl')
    val=load_jsonl(args.corpus/'validation.jsonl')
    review=load_jsonl(args.corpus/'review_queue.jsonl')
    examples=[*train,*val]
    summary=(validate_corpus_examples_parallel(examples,workers=args.workers) if args.deep
             else validate_corpus_examples(examples,deep=False))
    if summary!=manifest.get('summary'):
        raise SystemExit('Recomputed corpus summary does not match manifest')
    review_expected=manifest.get('review',{}).get('queue_count')
    review_summary=validate_review_queue(review,examples,expected_count=review_expected)
    if review_summary!=manifest.get('review',{}).get('summary'):
        raise SystemExit('Recomputed review summary does not match manifest')

    observed_files={}
    for name,items in [('train.jsonl',train),('validation.jsonl',val),('review_queue.jsonl',review)]:
        raw=(args.corpus/name).read_bytes(); expected=manifest['files'][name]
        observed={'records':len(items),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
        if observed!=expected:
            raise SystemExit(f'{name} does not match manifest')
        observed_files[name]=observed
    result={
        'schema':'model-laboratory-interpreter-training-validation-report',
        'schema_version':'1.0',
        'laboratory_version':__version__,
        'status':'PASS',
        'deep':args.deep,
        'corpus_manifest_file_sha256':hashlib.sha256(manifest_raw).hexdigest(),
        'corpus_manifest_canonical_sha256':declared_manifest_sha,
        'summary':summary,
        'review':review_summary,
        'files':observed_files,
    }
    text=json.dumps(result,indent=2,ensure_ascii=False)+'\n'
    if args.output is not None:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(text,encoding='utf-8')
    print(text,end='')
    return 0
if __name__=='__main__': raise SystemExit(main())
