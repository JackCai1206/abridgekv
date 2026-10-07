#!/usr/bin/env python3
"""Negative control for the serving parity check: would "attending to everything" be detected?

Replays the token sequence a vLLM CompressLM run produced through the SAME trained weights with plain causal attention over
the whole raw history (no pause tokens, no eviction; content positions are identical to the in-block layout's), and compares
the log-probability of each vLLM token with vLLM's own. If vLLM's numbers sit close to the compressing reference
(check_teacher_forced_parity.py) and far from this replay, the served model does not see evicted raw KV.

  python scripts/check_full_attention_control.py --model-dir <hf export> --vllm-result <run>.vllm.json [--out <json>] [--C 512]
"""
import argparse, json, os, torch
from transformers import AutoModelForCausalLM
ap = argparse.ArgumentParser(); ap.add_argument("--model-dir", required=True); ap.add_argument("--vllm-result", required=True)
ap.add_argument("--out", default=""); ap.add_argument("--C", type=int, default=512); a = ap.parse_args()
V = json.load(open(a.vllm_result)); prompt = V["prompt_token_ids"]; toks = V["output_token_ids"]; vtop = V["top_logprobs"]
m = AutoModelForCausalLM.from_pretrained(a.model_dir, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
ids = torch.tensor([prompt + toks], device="cuda")
with torch.inference_mode():
    logits = m(ids).logits[0, len(prompt) - 1: len(prompt) - 1 + len(toks)].float()
lp = torch.log_softmax(logits, -1)
rows = []
for i, t in enumerate(toks):
    v = next((d["logprob"] for d in vtop[i] if d["token_id"] == t), None)
    rows.append({"i": i, "abs": len(prompt) + i, "tok": t, "full_argmax": int(lp[i].argmax()), "full_lp_tok": float(lp[i, t]), "vllm_lp_tok": v})
P, C = len(prompt), a.C
def report(w, label):
    w = [r for r in w if r["vllm_lp_tok"] is not None]
    d = [abs(r["full_lp_tok"] - r["vllm_lp_tok"]) for r in w]; ag = sum(r["full_argmax"] == r["tok"] for r in w) / len(w)
    print(f"[full-attn] {label} n={len(w)}: mean |dlogprob| {sum(d)/len(d):.4f} argmax-agree {ag:.3f}"); return {"label": label, "n": len(w), "mean": sum(d) / len(d), "argmax_agree": ag}
summ = [report(rows, "all")]
summ.append(report([r for r in rows if r["abs"] < (P // C + 1) * C], "before the first compression"))
summ.append(report([r for r in rows if r["abs"] >= (P // C + 1) * C], "after the first compression"))
if a.out: json.dump({"summary": summ, "rows": rows}, open(a.out, "w"))
