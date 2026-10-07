#!/usr/bin/env python3
"""Per-token summary of a mask-control run (launch/mask_control.sh): for every token after the first compression,
is vLLM's log-prob of the chosen token closer to the compressing reference (A) or to full attention over the raw history (B)?
Restricted to "informative" tokens where A and B themselves differ by > 0.05 nats, so the near-certain tokens do not dilute it.
  python3 scripts/summarize_mask_control.py <maskcontrol dir> [...]"""
import json, sys, glob, os
for d in sys.argv[1:]:
    for f in sorted(glob.glob(os.path.join(d, "p*.tf.json"))):
        p = os.path.basename(f).split(".")[0]; g = os.path.join(d, p + ".full.json")
        if not os.path.exists(g): continue
        A = json.load(open(f))["rows"]; B = json.load(open(g))["rows"]; assert len(A) == len(B)
        rows = [(a, b) for a, b in zip(A, B) if a["abs"] >= 512 and a["vllm_lp_tok"] is not None]
        inf = [(a, b) for a, b in rows if abs(a["hf_lp_tok"] - b["full_lp_tok"]) > 0.05]
        closerA = sum(abs(a["vllm_lp_tok"] - a["hf_lp_tok"]) < abs(a["vllm_lp_tok"] - b["full_lp_tok"]) for a, b in inf)
        mA = sum(abs(a["vllm_lp_tok"] - a["hf_lp_tok"]) for a, b in inf) / max(len(inf), 1)
        mB = sum(abs(a["vllm_lp_tok"] - b["full_lp_tok"]) for a, b in inf) / max(len(inf), 1)
        print(f"{os.path.basename(d)} {p}: {len(rows)} tokens after the first compression, {len(inf)} where A and B differ by >0.05; "
              f"vLLM closer to the compressing reference on {closerA}/{len(inf)}; mean |vLLM-A| {mA:.4f} vs |vLLM-B| {mB:.4f}")
