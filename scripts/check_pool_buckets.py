#!/usr/bin/env python3
"""Guard against SFT-pool drift across clusters: load PAIRS, apply the trainer's filter (rebucket to C, 2 <= n_blocks <= NMAX,
drop the first EVALN held-out pairs) and compare the per-bucket counts with a reference `buckets={...}` line taken from a
trusted train.log (the canonical stream depends only on these counts + the pickle order, so equal counts + same builder
order => identical per-step example sets). Exit 3 on mismatch.
  PAIRS=<pkl> REF=<file with buckets={...}> [C=512 NMAX=32 EVALN=192] python3 scripts/check_pool_buckets.py"""
import os, re, sys
from collections import Counter
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from abridgekv.sft_quota_utils import rebucket_pairs

pairs_path, ref_path = os.environ["PAIRS"], os.environ["REF"]
C, NMAX, EVALN = int(os.environ.get("C", 512)), int(os.environ.get("NMAX", 32)), int(os.environ.get("EVALN", 192))
ref = re.search(r"buckets=\{[^}]*\}", open(ref_path).read())
assert ref, f"no buckets={{...}} line in {ref_path}"
ref = {int(k): int(v) for k, v in re.findall(r"(\d+):(\d+)", ref.group(0))}
kept = [p for p in rebucket_pairs(torch.load(pairs_path, weights_only=False), C) if 2 <= p["n_blocks"] <= NMAX]
mine = dict(Counter(p["n_blocks"] for p in kept[EVALN:]))
if mine == ref:
    print(f"[pool-check] OK {len(kept)} pairs, {len(mine)} buckets match {ref_path}")
else:
    diff = {n: (ref.get(n), mine.get(n)) for n in sorted(set(ref) | set(mine)) if ref.get(n) != mine.get(n)}
    print(f"[pool-check] MISMATCH vs {ref_path}: {{n: (ref, mine)}} = {diff}"); sys.exit(3)
