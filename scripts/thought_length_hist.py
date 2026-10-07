"""Thought-length statistics of the SFT pool split on "\n\n" (LightThinker's thought boundary).
Boundary = any token whose decoded string CONTAINS "\n\n" (Qwen merges ".\n\n" etc. into one token)."""
import os, sys, numpy as np, torch
from transformers import AutoTokenizer
os.environ["HF_HUB_OFFLINE"] = "1"
POOL = os.environ["POOL"]
NSAMP = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Base")
V = len(tok)
bnd = np.zeros(V, dtype=bool)
for i in range(V):
    s = tok.decode([i])
    if "\n\n" in s: bnd[i] = True
print(f"vocab {V}, boundary tokens (decode contains \\n\\n): {int(bnd.sum())}", flush=True)
ex = [tok.decode([i]) for i in np.nonzero(bnd)[0][:12]]
print("examples:", [repr(e) for e in ex], flush=True)
if True:
    pool = torch.load(POOL, weights_only=False)
rng = np.random.default_rng(0)
idx = rng.choice(len(pool), size=min(NSAMP, len(pool)), replace=False)
seg = []          # per-thought lengths (tokens), including the boundary token
per_ex = []       # (resp_len, n_thoughts)
for j in idx:
    r = np.asarray(pool[j]["response"], dtype=np.int64)
    b = np.nonzero(bnd[r])[0]
    cuts = np.concatenate([[-1], b, [len(r) - 1]]) if (len(b) == 0 or b[-1] != len(r) - 1) else np.concatenate([[-1], b])
    L = np.diff(cuts)
    seg.append(L); per_ex.append((len(r), len(L)))
seg = np.concatenate(seg); per_ex = np.asarray(per_ex)
q = np.percentile(seg, [5, 25, 50, 75, 90, 95, 99])
print(f"examples sampled {len(idx)}; thoughts {len(seg)}; mean resp {per_ex[:,0].mean():.0f} tok, mean thoughts/example {per_ex[:,1].mean():.1f}")
print(f"thought length tokens: mean {seg.mean():.1f}  p5/25/50/75/90/95/99 = {q.round(0).tolist()}  max {seg.max()}")
for K in (9, 7):
    for cap in (None, 128, 256, 512):
        L = seg if cap is None else np.minimum(seg, cap)
        nseg = len(L) if cap is None else int(np.ceil(seg / cap).sum())
        # steady-state ratio: raw tokens per cache row (each thought -> K rows); with a cap a long thought splits
        print(f"  K={K} cap={cap}: raw/cache = {seg.sum() / (K * nseg):.2f}:1   (thoughts>{cap}: {(seg > cap).mean()*100 if cap else 0:.1f}%)")
# fraction of thoughts shorter than K (compression would EXPAND)
print(f"thoughts shorter than 9 tokens: {(seg < 9).mean()*100:.1f}%  shorter than 36 (ratio<4): {(seg < 36).mean()*100:.1f}%")
