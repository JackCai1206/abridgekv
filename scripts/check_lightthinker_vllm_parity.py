"""LightThinker serving parity: vLLM (boundary-triggered compaction, shadow overlay) vs the trained forward.

Stage `vllm` (vLLM env, DEV_VLLM shadow on PYTHONPATH): greedy decode of N suite problems at batch N with debug
logprobs; the visible output keeps the sampled splitters, so the processed sequence is recoverable.
Stage `hf` (train env): rebuild the sequence the decoder processed (splitter -> 9 cache tokens -> continue, via
lightthinker.augment_generated), run ONE forward with the TRAINING mask (FlexAttention BlockMask, sequential
positions) and compare, at every generated token, the HF logprob of vLLM's token with vLLM's logprob, and the
HF argmax with vLLM's greedy token (teacher-forced along vLLM's own trajectory, so a near-tie flip cannot cascade).
A control forward with a plain causal mask (no eviction) must disagree clearly after the first thought: that
proves the check sees whether vLLM really evicted each thought.

  PYTHONPATH=$DEV_VLLM:. python scripts/check_lightthinker_vllm_parity.py --stage vllm --model-dir HF --out O.json
  python scripts/check_lightthinker_vllm_parity.py --stage hf --model-dir HF --out O.json
"""
import argparse
import json
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def stage_vllm(a):
    from pathlib import Path
    from vllm import LLM, SamplingParams
    from vllm.config.compresslm import CompressLMConfig
    from prompt_format import format_user_content

    meta = json.loads((Path(a.model_dir) / "compresslm_config.json").read_text())
    meta = {k: v for k, v in meta.items() if k in CompressLMConfig.__dataclass_fields__ and not k.startswith("_")}
    cfg = CompressLMConfig(**meta)
    if not cfg.uses_boundary_trigger:
        raise SystemExit("[lt-parity] the vLLM package on the path has no boundary trigger (DEV_VLLM shadow missing?)")
    cfg.debug = True                                   # CompressLM exposes logprobs only in debug mode
    llm = LLM(model=a.model_dir, dtype="bfloat16", max_model_len=a.max_model_len, max_num_seqs=a.n,
              gpu_memory_utilization=float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.85")), enable_prefix_caching=False,
              compresslm_config=cfg)
    tok = llm.get_tokenizer()
    probs = [json.loads(line) for line in open(a.problems)][: a.n]
    prompts = [tok.encode(format_user_content(p["problem"], True), add_special_tokens=True) for p in probs]
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens, skip_special_tokens=False,
                        bad_words=cfg.code_tokens, logprobs=a.top_logprobs)
    outs = llm.generate([{"prompt_token_ids": p} for p in prompts], sp, use_tqdm=False)
    rows = []
    for p, o in zip(prompts, outs):
        c = o.outputs[0]
        top = [[{"token_id": int(t), "logprob": float(lp.logprob)} for t, lp in d.items()] for d in c.logprobs]
        rows.append({"prompt_token_ids": p, "output_token_ids": list(c.token_ids), "top_logprobs": top,
                     "finish_reason": c.finish_reason})
    n_split = sum(r["output_token_ids"].count(cfg.boundary_token_id) for r in rows)
    print(f"[lt-parity] vLLM: {len(rows)} requests, {sum(len(r['output_token_ids']) for r in rows)} tokens, "
          f"{n_split} thoughts closed (splitters sampled)", flush=True)
    json.dump({"model_dir": a.model_dir, "rows": rows}, open(a.out, "w"))


def stage_hf(a):
    import numpy as np
    import torch
    from torch.nn.attention.flex_attention import create_block_mask
    from transformers import AutoModelForCausalLM
    from abridgekv.lightthinker import NEVER, augment_generated, lt_ids, mask_mod_factory, LTIds

    V = json.load(open(a.out))
    meta = json.load(open(os.path.join(a.model_dir, "compresslm_config.json")))
    ids = LTIds(cache=tuple(meta["code_token_ids"]), splitter=meta["boundary_token_id"], cont=meta["continue_token_id"])
    model = AutoModelForCausalLM.from_pretrained(a.model_dir, dtype=torch.bfloat16, attn_implementation="flex_attention").cuda().eval()

    def logprobs(seq, hide):
        x = torch.tensor(seq, device="cuda")[None]
        S = x.size(1)
        h = torch.tensor(hide, device="cuda")[None]
        bm = create_block_mask(mask_mod_factory(h), B=1, H=None, Q_LEN=S, KV_LEN=S, device="cuda")
        pos = torch.arange(S, device="cuda")[None]
        with torch.no_grad():
            hid = model.model(input_ids=x, position_ids=pos, attention_mask=bm).last_hidden_state[0]
            return hid

    stats = {"n": 0, "agree": 0, "dlp": [], "ctl_dlp": [], "first_div": [], "near_tie_margin": [],
             "dlp_after_continue": [], "dlp_in_thought": []}
    per_req = []
    for r in V["rows"]:
        gen = r["output_token_ids"]
        seq, hide, pred = augment_generated(r["prompt_token_ids"], gen, ids)
        hid = logprobs(seq, hide)
        hid_ctl = logprobs(seq, np.full_like(hide, int(NEVER)))       # control: nothing evicted
        with torch.no_grad():
            lp = torch.log_softmax(model.lm_head(hid[torch.tensor(pred, device="cuda")]).float(), -1)
            lpc = torch.log_softmax(model.lm_head(hid_ctl[torch.tensor(pred, device="cuda")]).float(), -1)
        first_div = None; n_agree = 0
        for j, t in enumerate(gen):
            v = next((d["logprob"] for d in r["top_logprobs"][j] if d["token_id"] == t), None)
            hf_arg = int(lp[j].argmax())
            n_agree += hf_arg == t
            if hf_arg != t and first_div is None:
                first_div = j
                top2 = lp[j].topk(2).values
                stats["near_tie_margin"].append(float(top2[0] - top2[1]))
            if v is None:
                continue
            d = abs(float(lp[j, t]) - v)
            stats["dlp"].append(d)
            stats["ctl_dlp"].append(abs(float(lpc[j, t]) - v))
            after = j > 0 and gen[j - 1] == ids.splitter
            (stats["dlp_after_continue"] if after else stats["dlp_in_thought"]).append(d)
        stats["n"] += len(gen); stats["agree"] += n_agree
        stats["first_div"].append(first_div)
        per_req.append({"tokens": len(gen), "thoughts": gen.count(ids.splitter), "argmax_agree": n_agree / max(1, len(gen)),
                        "first_divergence": first_div, "finish": r["finish_reason"]})

    def m(x):
        return float(np.mean(x)) if x else float("nan")
    ctl_after = m([c for c, d in zip(stats["ctl_dlp"], stats["dlp"])])
    summary = {
        "requests": len(V["rows"]), "tokens": stats["n"], "thoughts": sum(p["thoughts"] for p in per_req),
        "argmax_agree": stats["agree"] / max(1, stats["n"]),
        "exact_greedy_requests": sum(p["first_divergence"] is None for p in per_req),
        "mean_abs_dlogprob": m(stats["dlp"]), "max_abs_dlogprob": max(stats["dlp"]) if stats["dlp"] else float("nan"),
        "mean_abs_dlogprob_first_token_after_continue": m(stats["dlp_after_continue"]),
        "mean_abs_dlogprob_in_thought": m(stats["dlp_in_thought"]),
        "control_no_eviction_mean_abs_dlogprob": ctl_after,
        "first_divergence_top2_margins": stats["near_tie_margin"],
        "per_request": per_req,
    }
    ok = (summary["argmax_agree"] >= 0.99 and summary["mean_abs_dlogprob"] < 0.05
          and summary["control_no_eviction_mean_abs_dlogprob"] > 3 * summary["mean_abs_dlogprob"]
          and summary["thoughts"] > 0)
    summary["pass"] = bool(ok)
    json.dump(summary, open(a.out.replace(".json", ".summary.json"), "w"), indent=1)
    print(f"[lt-parity] {summary['requests']} requests, {summary['tokens']} tokens, {summary['thoughts']} thoughts; "
          f"argmax agree {summary['argmax_agree']:.4f}; exact-greedy requests {summary['exact_greedy_requests']}/{summary['requests']}; "
          f"mean|dlogprob| {summary['mean_abs_dlogprob']:.4f} (first token after continue "
          f"{summary['mean_abs_dlogprob_first_token_after_continue']:.4f}, in thought {summary['mean_abs_dlogprob_in_thought']:.4f}) "
          f"max {summary['max_abs_dlogprob']:.3f}; control without eviction {summary['control_no_eviction_mean_abs_dlogprob']:.4f}; "
          f"first-divergence top-2 margins {['%.3f' % x for x in summary['first_divergence_top2_margins']]}")
    print("[lt-parity] PASS" if ok else "[lt-parity] FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["vllm", "hf"], required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--problems", default="data/math_eval/aime2025.jsonl")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--top-logprobs", type=int, default=5)
    a = ap.parse_args()
    (stage_vllm if a.stage == "vllm" else stage_hf)(a)
