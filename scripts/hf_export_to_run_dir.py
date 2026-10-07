#!/usr/bin/env python3
"""Rebuild an sft_ddp-style run directory (model.pt + metadata) from a published CompressLM HF export, so a checkpoint
that only exists on the Hub can go through the evaluation scripts unchanged.

Reverse of scripts/convert_compresslm_checkpoint_to_hf.py for the plain/uniform export: the export's embedding is the
base vocab (rows < first code id) followed by the K exported code rows (`pause`); the base model's unused padding rows
above the real vocab are restored from the base checkpoint so shapes match CompressLM(base). `copy_token` is not in an
export and stays absent (only used with replay_tail > 0; sft_ddp/parity load with strict=False). Weights stay bf16.

  python3 scripts/hf_export_to_run_dir.py --export <dir> --out $OUTPUT_BASE/sft_ddp/<run> [--step 936] [--held-out-ce X]
"""
import argparse, json, os, sys
import torch
from transformers import AutoModelForCausalLM
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

ap = argparse.ArgumentParser()
ap.add_argument("--export", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--base", default="Qwen/Qwen3-4B-Base"); ap.add_argument("--step", type=int, default=0)
ap.add_argument("--held-out-ce", type=float, default=None); ap.add_argument("--source", default="")
a = ap.parse_args()

cfg = json.load(open(os.path.join(a.export, "compresslm_config.json")))
K = int(cfg["num_code_tokens"]); code_ids = cfg["code_token_ids"]; first = code_ids[0]
assert code_ids == list(range(first, first + K)), "code ids must be contiguous"
idx = json.load(open(os.path.join(a.export, "model.safetensors.index.json")))["weight_map"]
sd = {}
for f in sorted(set(idx.values())):
    sd.update(load_file(os.path.join(a.export, f)))
emb = sd.pop("model.embed_tokens.weight")
assert emb.shape[0] == first + K, (emb.shape, first, K)
pause = emb[first:first + K].clone()

base_dir = snapshot_download(a.base, local_files_only=True)
base = AutoModelForCausalLM.from_pretrained(base_dir, torch_dtype=torch.bfloat16, local_files_only=True)
bemb = base.get_input_embeddings().weight.detach()
V = bemb.shape[0]; assert V >= first + K
full = bemb.clone(); full[:first] = emb[:first]          # trained rows; padding rows above the real vocab stay as in the base
state = {"model." + k: v for k, v in sd.items()}
state["model.model.embed_tokens.weight"] = full
if getattr(base.config, "tie_word_embeddings", False) and "model.lm_head.weight" not in state:
    state["model.lm_head.weight"] = full
state["pause"] = pause
# shape check against a real CompressLM shell
sys.path.insert(0, os.getcwd())
from abridgekv.compress_forward import CompressLM
shell = CompressLM(AutoModelForCausalLM.from_config(base.config), int(cfg["block_size"]), N=2, K=K, L=int(cfg["num_passes"]),
                   pause_xattn=cfg.get("pause_xattn", "full"))
want = {k: tuple(v.shape) for k, v in shell.state_dict().items()}
bad = [k for k, v in state.items() if k not in want or tuple(v.shape) != want[k]]
assert not bad, bad[:5]
missing = sorted(set(want) - set(state)); assert set(missing) <= {"copy_token"}, missing
quotas = [[K] * 8 for _ in range(36)]
payload = {"model": state, "model_format": "full", "C": int(cfg["block_size"]), "K": K, "L": int(cfg["num_passes"]),
           "pause_xattn": cfg.get("pause_xattn", "full"), "head_code_quotas": quotas, "head_code_quota_budget": str(K),
           "raw_prompt": int(bool(cfg.get("raw_prompt", True))), "raw_tail": int(cfg.get("raw_tail", 0)),
           "replay_tail": int(cfg.get("replay_tail", 0)), "retain_tail_kv": int(cfg.get("retain_tail_kv", 0)),
           "retain_tail_policy": cfg.get("retain_tail_policy", "rolling"),
           "code_exclude_replay_tail": int(bool(cfg.get("code_exclude_replay_tail", False))), "step": a.step,
           "code_positions": cfg.get("code_positions", "appended"), "base": a.base, "arm": "compressed", "cross_block_loss": 1,
           "seg_pool": 0, "gated_seg_pool": 0, "lora_rank": 0, "source_export": a.source or a.export,
           "note": "rebuilt from an HF export by scripts/hf_export_to_run_dir.py; bf16 weights; copy_token absent"}
os.makedirs(a.out, exist_ok=True)
torch.save(payload, os.path.join(a.out, "model.pt"))
open(os.path.join(a.out, "model.pt.complete"), "w").write("ok\n")
json.dump(quotas, open(os.path.join(a.out, "trained_schedule.json"), "w"))
rec = {"step": a.step, "phase": "final", "head_code_quotas": quotas}
if a.held_out_ce is not None: rec["held_out_CE"] = a.held_out_ce
open(os.path.join(a.out, "heldout_position_ce.jsonl"), "w").write(json.dumps(rec) + "\n")
open(os.path.join(a.out, "train.log"), "a").write(f"[rebuild] run dir rebuilt from {a.source or a.export} (bf16); phase=final step={a.step}\n")
print(f"[rebuild] wrote {a.out}/model.pt ({os.path.getsize(os.path.join(a.out,'model.pt'))/2**30:.2f} GiB), K={K}, "
      f"retain_tail_kv={payload['retain_tail_kv']}/{payload['retain_tail_policy']}, code_positions={payload['code_positions']}")
