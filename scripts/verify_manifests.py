#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def check(base,manifest_name):
    obj=json.load((base/manifest_name).open(encoding='utf-8'))
    errors=[]
    for item in obj['files']:
        p=base/item['path']
        if not p.is_file(): errors.append((item['path'],'missing')); continue
        if p.stat().st_size!=item['size_bytes']: errors.append((item['path'],'size'))
        if sha(p)!=item['sha256']: errors.append((item['path'],'sha256'))
    assert not errors,errors
    return len(obj['files'])

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',default=None); a=p.parse_args()
    root=Path(a.root).resolve() if a.root else Path(__file__).resolve().parents[1]
    n1=check(root,'MANIFEST.json')
    n2=check(root/'CODE_APPENDIX','CODE_APPENDIX_MANIFEST.json')
    n3=check(root/'DATA_APPENDIX','DATA_APPENDIX_MANIFEST.json')
    print(f'[OK] Manifests verified: root={n1}, code={n2}, data={n3}.')
if __name__=='__main__':main()
