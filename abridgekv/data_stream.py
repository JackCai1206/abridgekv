"""Hardware-independent training data stream for sft_ddp.py.

The native sampler in sft_ddp.py draws, per microbatch, a bucket N from a shared generator and `micro` examples per
rank from a rank-seeded generator. The resulting stream therefore depends on (world, micro, accum): the archived
headline runs (world=8, micro=1, accum=32) and a 4-GPU run (micro=4/8) see different examples even at the same seed.

CanonicalStream emulates the native sampler of a fixed *virtual* configuration (default 8x1x32 = the headline runs)
on any hardware and returns, per optimizer step, exactly the multiset of (N, example) draws that configuration
produced. pack_draws then regroups those draws into same-N microbatches of at most `micro` examples and assigns them
deterministically to the physical ranks. The step loss is a token-weighted mean over the whole step, so the grouping
does not change the gradient in exact arithmetic; only the example set per step matters, and that is now identical
across machines.
"""
from __future__ import annotations

import torch


class CanonicalStream:
    def __init__(self, buckets, seed, vworld=8, vmicro=1, vaccum=32):
        self.buckets = buckets
        self.bkeys = [n for n in sorted(buckets) if len(buckets[n]) >= vmicro]   # native: buckets with >= micro examples
        assert self.bkeys, "no bucket has >= vmicro examples"
        self.g_shape = torch.Generator().manual_seed(seed)
        self.g_data = [torch.Generator().manual_seed(seed * 100003 + r + 1) for r in range(vworld)]
        self.vworld, self.vmicro, self.vaccum = vworld, vmicro, vaccum
        self.steps = 0

    @property
    def draws_per_step(self):
        return self.vworld * self.vmicro * self.vaccum

    def _draw(self, materialize):
        draws = []
        for _ in range(self.vaccum):
            n = self.bkeys[torch.randint(len(self.bkeys), (1,), generator=self.g_shape).item()]   # SAME on every virtual rank
            L = len(self.buckets[n])
            for g in self.g_data:                                                                    # each virtual rank's own RNG
                idx = torch.randperm(L, generator=g)[: self.vmicro]
                if materialize:
                    draws.extend((n, self.buckets[n][i]) for i in idx.tolist())
        self.steps += 1
        return draws

    def draw_step(self):
        """All (N, example) draws of the next optimizer step, in virtual (microbatch, rank, example) order."""
        return self._draw(True)

    def skip(self, steps):
        """Advance past `steps` optimizer steps (exact RNG replay, nothing materialized)."""
        for _ in range(int(steps)):
            self._draw(False)


def pack_draws(draws, micro, world, rank, cost=lambda n, size: size * n * n):
    """Group one step's draws into same-N microbatches of <= micro examples and assign them to ranks.

    Deterministic given the input, so every rank computes the same plan and takes its own share. Chunks are assigned
    longest-processing-time first to the least-loaded rank (cost ~ examples * N^2, the attention-dominated step cost);
    every rank gets at least one microbatch (required: each rank must run exactly one gradient-synchronising backward
    per step). Returns [(n, [examples...]), ...] for `rank`."""
    by_n = {}
    for n, ex in draws:
        by_n.setdefault(n, []).append(ex)
    chunks = []
    for n in sorted(by_n):
        items = by_n[n]
        for i in range(0, len(items), micro):
            chunks.append((n, items[i : i + micro]))
    assert len(chunks) >= world, f"only {len(chunks)} microbatches for {world} ranks; lower --micro"
    order = sorted(range(len(chunks)), key=lambda i: (-cost(chunks[i][0], len(chunks[i][1])), i))
    loads = [0.0] * world
    assign = [[] for _ in range(world)]
    for k, i in enumerate(order):
        r = k if k < world else min(range(world), key=lambda r: (loads[r], r))   # first `world` chunks seed one per rank
        assign[r].append(i)
        loads[r] += cost(chunks[i][0], len(chunks[i][1]))
    return [chunks[i] for i in sorted(assign[rank])]


def parse_stream_spec(spec):
    """'canonical' -> (8,1,32); 'canonical:8x1x32' -> (8,1,32); anything else -> None (native sampler)."""
    if not spec or spec == "native":
        return None
    head, _, tail = spec.partition(":")
    if head != "canonical":
        raise ValueError(f"unknown data stream spec {spec!r}")
    vw, vm, va = (int(x) for x in (tail or "8x1x32").split("x"))
    return vw, vm, va
