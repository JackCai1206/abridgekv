"""Small, dependency-free LoRA utilities for CompressLM training."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn


DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


class LoRALinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"LoRA dropout must be in [0, 1), got {dropout}")
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / rank
        self.dropout = nn.Dropout(dropout)
        for parameter in self.base_layer.parameters():
            parameter.requires_grad_(False)
        factory = {
            "device": base_layer.weight.device,
            "dtype": base_layer.weight.dtype,
        }
        self.lora_A = nn.Parameter(
            torch.empty(rank, base_layer.in_features, **factory)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(base_layer.out_features, rank, **factory)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def in_features(self):
        return self.base_layer.in_features

    @property
    def out_features(self):
        return self.base_layer.out_features

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    def forward(self, inputs):
        base = self.base_layer(inputs)
        adapter = F.linear(
            F.linear(self.dropout(inputs), self.lora_A),
            self.lora_B,
        )
        return base + adapter.to(base.dtype) * self.scaling

    def merged_weight(self):
        update = self.lora_B.float() @ self.lora_A.float()
        return self.base_layer.weight.float() + update * self.scaling


def inject_lora(
    module: nn.Module,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    target_modules: Iterable[str] = DEFAULT_TARGET_MODULES,
):
    targets = frozenset(target_modules)
    replaced = []

    def visit(parent: nn.Module, prefix: str):
        for name, child in list(parent.named_children()):
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and name in targets:
                setattr(
                    parent,
                    name,
                    LoRALinear(child, rank, alpha, dropout),
                )
                replaced.append(path)
            else:
                visit(child, path)

    visit(module, "")
    if not replaced:
        raise ValueError(
            "LoRA did not find any target Linear modules among "
            f"{sorted(targets)}"
        )
    return replaced


def configure_lora_trainability(
    core: nn.Module,
    rank: int,
    alpha: float,
    dropout: float,
    target_modules: Iterable[str],
    compressed: bool,
):
    for parameter in core.parameters():
        parameter.requires_grad_(False)
    replaced = inject_lora(
        core.model,
        rank,
        alpha,
        dropout,
        target_modules,
    )
    if compressed:
        for name, parameter in core.named_parameters():
            if not name.startswith("model."):
                parameter.requires_grad_(True)
    return replaced


def trainable_state_dict(module: nn.Module):
    trainable = {
        name for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }
    return {
        name: value
        for name, value in module.state_dict().items()
        if name in trainable
    }


def remap_full_state_for_lora(module: nn.Module, state):
    """Map a normal Linear state dict into an already LoRA-wrapped module."""
    remapped = dict(state)
    for name, child in module.named_modules():
        if not isinstance(child, LoRALinear):
            continue
        for suffix in ("weight", "bias"):
            source = f"{name}.{suffix}"
            target = f"{name}.base_layer.{suffix}"
            if source in remapped:
                remapped[target] = remapped.pop(source)
    return remapped


def merged_state_dict(module: nn.Module):
    """Return a regular, LoRA-free state dict with adapter deltas merged."""
    state = module.state_dict()
    wrappers = {
        name: child
        for name, child in module.named_modules()
        if isinstance(child, LoRALinear)
    }
    merged = {}
    for name, value in state.items():
        if any(
            name == prefix or name.startswith(f"{prefix}.")
            for prefix in wrappers
        ):
            continue
        merged[name] = value
    for name, child in wrappers.items():
        merged[f"{name}.weight"] = child.merged_weight().to(
            child.base_layer.weight.dtype
        )
        if child.base_layer.bias is not None:
            merged[f"{name}.bias"] = child.base_layer.bias
    return merged
