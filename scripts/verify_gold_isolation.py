#!/usr/bin/env python3
from __future__ import annotations
import argparse, ast
from pathlib import Path

BAD_TOKENS=('final_test_gold','test_gold','offline_reference','reference.json')

def string_literals(path):
    try: tree=ast.parse(path.read_text(encoding='utf-8'))
    except Exception: return []
    return [x.value for x in ast.walk(tree) if isinstance(x,ast.Constant) and isinstance(x.value,str)]

def main():
    p=argparse.ArgumentParser(); p.add_argument('--appendix',default=None); a=p.parse_args()
    release=Path(__file__).resolve().parents[1]
    appendix=Path(a.appendix).resolve() if a.appendix else release/'CODE_APPENDIX'
    bases=[appendix/'framework_src',appendix/'02_FROZEN_INFERENCE',appendix/'01_DEV39_WORKSPACE'/'policy_distillation'/'scripts']
    for base in bases: assert base.is_dir(),base
    violations=[]
    for base in bases:
        for q in base.rglob('*.py'):
            for s in string_literals(q):
                low=s.lower()
                if any(t in low for t in BAD_TOKENS):
                    if q.name != 'verify_stage1_integrity.py':
                        violations.append((str(q.relative_to(appendix)),s[:160]))
    assert not violations,violations
    print('[OK] Inference source contains no direct test-gold/reference path literals.')
if __name__=='__main__': main()
