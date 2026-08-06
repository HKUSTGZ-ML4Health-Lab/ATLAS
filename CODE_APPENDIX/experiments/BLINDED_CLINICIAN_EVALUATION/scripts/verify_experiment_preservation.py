#!/usr/bin/env python3
# AAAI-27 paper reference: Blinded Clinician Evaluation isolation guarantee.
"""Verify that clinician-evaluation additions leave all reported experiments unchanged."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--appendix-root", required=True)
    p.add_argument("--guard", required=True)
    args = p.parse_args()
    root = Path(args.appendix_root).resolve()
    guard = json.loads(Path(args.guard).read_text(encoding="utf-8"))
    errors = []
    for item in guard["files"]:
        q = root / item["path"]
        if not q.is_file():
            errors.append((item["path"], "missing"))
            continue
        if q.stat().st_size != int(item["size_bytes"]):
            errors.append((item["path"], "size"))
        if sha256(q) != item["sha256"]:
            errors.append((item["path"], "sha256"))
    if errors:
        raise AssertionError(errors)
    print(f"[OK] {len(guard['files'])} experiment-critical artifacts remain byte-identical.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
