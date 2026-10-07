#!/usr/bin/env python3
"""Teacher-forced HF-vs-vLLM parity across many block boundaries.

Greedy parity stops being informative at the first near-tie flip (bf16 kernels differ between backends), so instead the HF
reference generator (CompressGen, proven equal to the trainer) REPLAYS the token sequence vLLM produced and we compare, at every
position, the log-probability each backend assigns to vLLM's chosen token (vLLM's number comes from its top-k logprobs of the
parity run). A serving bug that appears from the second compressed block onward shows up as a jump of |dlogprob| after a
boundary that does not appear in the uniform control.

  python scripts/check_teacher_forced_parity.py --checkpoint <model.pt> --vllm-result <parity>.vllm.json --schedule <quotas.json> [--out <json>]
"""
import argparse, json, os, sys, math, torch
sys.path.insert(0, os.getcwd())

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True); ap.add_argument("--vllm-result", required=True); ap.add_argument("--schedule", default="")
ap.add_argument("--base-model", default="Qwen/Qwen3-4B-Base"); ap.add_argument("--out", default=""); ap.add_argument("--C", type=int, default=512)
ap.add_argument("--skip-unused-codes", action="store_true", help="HF reference with skip-when-unused code rows (CompressLM.set_skip_unused_codes)")
a = ap.parse_args()
from huggingface_hub import snapshot_download
from abridgekv.gen_compress import CompressGen, _load_compresslm

V = json.load(open(a.vllm_result)); prompt = V["prompt_token_ids"]; toks = V["output_token_ids"]; vtop = V["top_logprobs"]
quotas = None
if a.schedule:
    q = json.load(open(a.schedule)); q = q.get("quotas", q) if isinstance(q, dict) else q; quotas = tuple(tuple(int(x) for x in r) for r in q)
local_base = snapshot_download(a.base_model, local_files_only=True)
clm, _, _, _ = _load_compresslm(a.checkpoint, local_base, "cuda", torch.bfloat16)
gen = CompressGen(clm, "cuda", raw_prompt_len=len(prompt) if clm.raw_prompt else 0, head_code_quotas=quotas, skip_unused_codes=a.skip_unused_codes)
rows = []
with torch.inference_mode():
    logits = gen.prefill(prompt)
    for i, t in enumerate(toks):
        lp = torch.log_softmax(logits.float(), -1)
        hf_tok = int(lp.argmax()); hf_lp_t = float(lp[t]); hf_top = float(lp[hf_tok])
        v_lp_t = next((d["logprob"] for d in vtop[i] if d["token_id"] == t), None)
        rows.append({"i": i, "abs": len(prompt) + i, "tok": t, "hf_argmax": hf_tok, "hf_lp_tok": hf_lp_t, "hf_lp_top": hf_top, "vllm_lp_tok": v_lp_t})
        if i + 1 < len(toks): logits = gen.step(t)
P = len(prompt); C = a.C
d = [abs(r["hf_lp_tok"] - r["vllm_lp_tok"]) for r in rows if r["vllm_lp_tok"] is not None]
agree = sum(r["hf_argmax"] == r["tok"] for r in rows) / len(rows)
print(f"[tf-parity] {os.path.basename(a.vllm_result)}: {len(rows)} replayed tokens, prompt {P}, blocks completed {(P + len(rows)) // C}; HF argmax == vLLM token {agree:.4f}; mean |dlogprob| {sum(d)/len(d):.4f} max {max(d):.4f}")
print("[tf-parity] per compressed-block window (abs positions), mean/max |dlogprob| and argmax agreement:")
summ = []
for b in range(P // C, (P + len(rows)) // C + 1):
    w = [r for r in rows if b * C <= r["abs"] < (b + 1) * C and r["vllm_lp_tok"] is not None]
    if not w: continue
    dd = [abs(r["hf_lp_tok"] - r["vllm_lp_tok"]) for r in w]; ag = sum(r["hf_argmax"] == r["tok"] for r in w) / len(w)
    first = [r for r in w if r["abs"] < b * C + 32]; fd = [abs(r["hf_lp_tok"] - r["vllm_lp_tok"]) for r in first]
    summ.append({"block": b, "n": len(w), "history_blocks": b, "mean": sum(dd) / len(dd), "max": max(dd), "argmax_agree": ag, "first32_mean": (sum(fd) / len(fd)) if fd else None})
    print(f"[tf-parity]   block {b:2d} (history {b} compressed blocks) n={len(w):3d}: mean {sum(dd)/len(dd):.4f} max {max(dd):.4f} argmax-agree {ag:.3f} first-32-after-boundary mean {(sum(fd)/len(fd)) if fd else float('nan'):.4f}")
if a.out: json.dump({"summary": summ, "rows": rows}, open(a.out, "w"))
