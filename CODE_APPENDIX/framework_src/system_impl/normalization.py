from __future__ import annotations
import re

def norm(s):
    s=str(s or '').lower().replace('≥',' >= ').replace('≤',' <= ')
    s=re.sub(r'[^a-z0-9]+',' ',s)
    return re.sub(r'\s+',' ',s).strip()

def phrase(text, pat):
    t=' '+norm(text)+' '; p=norm(pat)
    return bool(p) and (' '+p+' ') in t

def any_phrase(text, patterns): return any(phrase(text,p) for p in patterns or [])

def nset(xs): return {norm(x) for x in (xs or []) if norm(x)}
