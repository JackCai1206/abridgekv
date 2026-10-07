"""Muon optimizer (Newton-Schulz orthogonalized momentum) + hybrid Muon/AdamW param split.
Recipe: Moonshot "Muon is Scalable" + Keller Jordan modded-nanogpt.
  - 2D hidden matrices -> Muon; embeddings / lm_head / norms / 1D / scalars -> AdamW.
  - momentum 0.95 Nesterov; Newton-Schulz 5 steps quintic (3.4445,-4.7750,2.0315), bf16.
  - per-matrix update scale 0.2*sqrt(max(fan_out,fan_in)) -> matches AdamW update RMS -> reuse AdamW lr.
  - weight decay 0.1 shared across both.
"""
import torch
from torch import Tensor


@torch.no_grad()
def _zeropower_via_newtonschulz5(G: Tensor, steps: int = 5) -> Tensor:
    """Orthogonalize G (…, m, n) via 5 quintic Newton-Schulz iterations, run in bf16."""
    assert G.ndim >= 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)          # spectral norm <= 1
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    """Muon on 2D matrices. Assumes gradients are already all-reduced (standard DDP)."""
    def __init__(self, params, lr=2e-3, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.1):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      ns_steps=ns_steps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for grp in self.param_groups:
            mom, nest, ns, wd, lr = grp["momentum"], grp["nesterov"], grp["ns_steps"], grp["weight_decay"], grp["lr"]
            for p in grp["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                buf = st.get("momentum_buffer")
                if buf is None:
                    buf = st["momentum_buffer"] = torch.zeros_like(g)
                buf.mul_(mom).add_(g)
                g = g.add(buf, alpha=mom) if nest else buf
                g2 = g.reshape(g.size(0), -1)                       # flatten conv-like; matrices are already 2D
                u = _zeropower_via_newtonschulz5(g2, ns).view_as(p)
                fan_out, fan_in = p.size(0), p.numel() // p.size(0)
                scale = 0.2 * (max(fan_out, fan_in) ** 0.5)         # update-RMS match -> reuse AdamW lr
                if wd:
                    p.mul_(1 - lr * wd)
                p.add_(u.to(p.dtype), alpha=-lr * scale)


def build_muon_adamw(model, muon_lr=2e-3, adam_lr=3e-4, weight_decay=0.1,
                     adam_betas=(0.9, 0.95), muon_momentum=0.95, ns_steps=5):
    """Split params: 2D hidden matrices -> Muon; embeddings/lm_head/norms/biases/1D/scalars -> AdamW.
    Returns (muon_opt, adamw_opt). Step both, zero_grad both."""
    muon_params, adam_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_2d = p.ndim == 2
        is_embed_or_head = ("embed" in name.lower()) or ("lm_head" in name.lower()) or ("wte" in name.lower()) or ("wpe" in name.lower())
        if is_2d and not is_embed_or_head:
            muon_params.append(p)
        else:
            adam_params.append(p)                                  # embeds, head, RMSNorm gains, biases, scalars
    muon = Muon(muon_params, lr=muon_lr, momentum=muon_momentum, ns_steps=ns_steps, weight_decay=weight_decay)
    adamw = torch.optim.AdamW(adam_params, lr=adam_lr, betas=adam_betas, weight_decay=weight_decay, fused=True)
    n_m = sum(p.numel() for p in muon_params); n_a = sum(p.numel() for p in adam_params)
    print(f"[muon] {len(muon_params)} matrices ({n_m/1e6:.1f}M params) on Muon; "
          f"{len(adam_params)} tensors ({n_a/1e6:.1f}M) on AdamW")
    return muon, adamw
