"""Export an SFT ``CompressLM`` checkpoint for vanilla or pause-code vLLM."""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer


EXPORT_MANIFEST = "compresslm_export.json"


def checkpoint_file(path: str) -> Path:
    resolved = Path(path)
    if resolved.is_dir():
        resolved = resolved / "model.pt"
    if not resolved.is_file():
        raise FileNotFoundError(f"checkpoint not found: {resolved}")
    return resolved


def local_model_path(model: str) -> str:
    path = Path(model)
    if path.is_dir():
        return str(path)
    return snapshot_download(model, local_files_only=True)


def checkpoint_identity(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "checkpoint_size": stat.st_size,
        "checkpoint_mtime_ns": stat.st_mtime_ns,
    }


def export_is_current(
    output: Path,
    checkpoint: Path,
    arm: str,
    base_model: str,
) -> bool:
    try:
        manifest = json.loads((output / EXPORT_MANIFEST).read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    expected = checkpoint_identity(checkpoint)
    if any(manifest.get(key) != value for key, value in expected.items()):
        return False
    if arm != "auto" and manifest.get("arm") != arm:
        return False
    return not base_model or manifest.get("base_model") == base_model


def saved_embedding_rows(
    output: Path,
    token_ids: list[int],
) -> torch.Tensor:
    from safetensors import safe_open

    key = "model.embed_tokens.weight"
    index_path = output / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        shard = output / index["weight_map"][key]
    else:
        shard = output / "model.safetensors"
    with safe_open(shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)[token_ids]


def tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint16).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-model", default="")
    parser.add_argument(
        "--arm", choices=["auto", "compressed", "baseline"], default="auto"
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--code-rows", type=int, default=0,
                        help="export only the first N code rows (a UNIFORM per-head quota N of a K_max checkpoint): exact, since "
                             "rows >= N are never attended under nested-prefix quotas; lets vLLM serve the model on the plain "
                             "(non-packed) path with num_code_tokens=N")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    ckpt_path = checkpoint_file(args.checkpoint)
    output = Path(args.output)
    # Parallel eval shards export the same checkpoint at once; without a lock a shard can read a half-written
    # safetensors index (vLLM JSONDecodeError at engine start, vanilla eval 2077 shard 2). Held until exit.
    import fcntl
    output.parent.mkdir(parents=True, exist_ok=True)
    _lock = open(str(output) + ".lock", "w")
    fcntl.flock(_lock, fcntl.LOCK_EX)
    if args.verify_only:
        if not export_is_current(
            output, ckpt_path, args.arm, args.base_model
        ):
            raise SystemExit(1)
        print(f"[export] verified current export {output}")
        return
    if export_is_current(output, ckpt_path, args.arm, args.base_model):
        print(f"[export] verified current export {output} (written by a concurrent shard)")
        return

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(
            "expected an sft_ddp checkpoint payload containing model + metadata"
        )
    state = payload["model"]
    if not isinstance(state, dict):
        raise TypeError("checkpoint 'model' entry is not a state_dict")

    arm = payload.get("arm")
    if args.arm != "auto":
        arm = args.arm
    if arm not in ("compressed", "baseline"):
        raise ValueError(f"cannot infer checkpoint arm from {arm!r}")
    if arm == "compressed" and (
        bool(payload.get("seg_pool", 0))
        or bool(payload.get("gated_seg_pool", 0))
        or bool(payload.get("pool_init", 0))
        or "seg_pool_gate" in state
        or any(
            key == "pool_q"
            or key.startswith("pool_k.")
            or key.startswith("pool_v.")
            for key in state
        )
    ):
        raise ValueError(
            "dynamic content-dependent code initialization cannot be exported "
            "as static vLLM code-token embeddings"
        )
    checkpoint_base = payload.get("base")
    if (
        args.base_model
        and checkpoint_base
        and args.base_model != checkpoint_base
    ):
        raise ValueError(
            f"--base-model={args.base_model!r} conflicts with checkpoint "
            f"base={checkpoint_base!r}"
        )
    base_model = args.base_model or checkpoint_base
    if not base_model:
        raise ValueError("--base-model is required when checkpoint has no base metadata")

    output.mkdir(parents=True, exist_ok=True)
    (output / EXPORT_MANIFEST).unlink(missing_ok=True)
    dtype = getattr(torch, args.dtype)

    print(f"[export] loading {base_model} for {arm} checkpoint {ckpt_path}")
    model_path = local_model_path(base_model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        attn_implementation="sdpa",
        local_files_only=True,
    )

    base_state = {
        key[len("model.") :]: value
        for key, value in state.items()
        if key.startswith("model.")
    }
    if not base_state:
        raise ValueError("checkpoint contains no CompressLM model.* parameters")
    missing, unexpected = model.load_state_dict(base_state, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected base-model keys: {unexpected[:10]}")
    print(f"[export] base weights loaded; missing={len(missing)}")

    metadata = {
        "enable": arm == "compressed",
        "block_size": int(payload.get("C", 512)),
        "num_code_tokens": int(payload.get("K", 128)),
        "num_passes": int(payload.get("L", 1)),
        "pause_xattn": payload.get("pause_xattn", "full"),
        "raw_prompt": bool(payload.get("raw_prompt", 0)),
        "raw_tail": int(payload.get("raw_tail", 0)),
        "replay_tail": int(payload.get("replay_tail", 0)),
        "code_positions": str(payload.get("code_positions", "appended")),
        "retain_tail_kv": int(payload.get("retain_tail_kv", 0)),
        "retain_tail_policy": payload.get(
            "retain_tail_policy", "rolling"
        ),
        "code_exclude_replay_tail": bool(
            payload.get("code_exclude_replay_tail", False)
        ),
        "code_token_format": "<|compress_code_{:03d}|>",
    }

    if arm == "compressed":
        pause = state.get("pause")
        if pause is None:
            raise ValueError("compressed checkpoint is missing learned pause embeddings")
        if pause.shape[0] != metadata["num_code_tokens"]:
            raise ValueError(
                f"pause rows={pause.shape[0]} != K={metadata['num_code_tokens']}"
            )
        if args.code_rows:
            if not 0 < args.code_rows <= pause.shape[0]:
                raise ValueError(f"--code-rows {args.code_rows} must be in [1, K={pause.shape[0]}]")
            if args.code_rows != pause.shape[0]:
                if metadata["code_positions"] != "bitrev":
                    raise ValueError("--code-rows < K requires code_positions=bitrev (K-independent code positions)")
                print(f"[export] truncating code rows K={pause.shape[0]} -> {args.code_rows} (uniform quota, nested prefix)")
            pause = pause[: args.code_rows]
            metadata["num_code_tokens"] = int(args.code_rows)
            metadata["code_rows_of_k_max"] = int(pause.shape[0])
        code_tokens = [
            metadata["code_token_format"].format(index)
            for index in range(metadata["num_code_tokens"])
        ]
        added = tokenizer.add_special_tokens(
            {"additional_special_tokens": code_tokens}
        )
        if added != len(code_tokens):
            raise ValueError(
                f"expected to add {len(code_tokens)} code tokens, added {added}"
            )
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
        code_id_tensor = torch.tensor(code_ids, dtype=torch.long)
        with torch.no_grad():
            embedding_weight = model.get_input_embeddings().weight
            embedding_weight.index_copy_(
                0,
                code_id_tensor.to(embedding_weight.device),
                pause.to(
                    device=embedding_weight.device,
                    dtype=embedding_weight.dtype,
                ),
            )
        metadata["code_token_ids"] = code_ids
        print(
            f"[export] installed {len(code_ids)} learned code embeddings "
            f"at IDs [{min(code_ids)}, {max(code_ids)}]"
        )

    model.config.compresslm_config = metadata
    if arm == "compressed" and metadata.get("code_positions") == "appended":
        # Appended code rows consume RoPE positions: a 32K generation reaches ~32K * (C + K) / C positions, past the
        # base model's max_position_embeddings; vLLM sizes its rotary cos/sin cache from that field and a longer
        # sequence trips a device-side index assert (seen 2026-09-21 on the gist arm at 32K). Extend the cache; the
        # rotary values at positions below the original limit are unchanged.
        base_max = int(getattr(model.config, "max_position_embeddings", 32768))
        stretch = (metadata["block_size"] + metadata["num_code_tokens"]) / metadata["block_size"]
        need = int(base_max * stretch) + 1024
        new_max = 1 << (need - 1).bit_length()
        if new_max > base_max:
            model.config.max_position_embeddings = new_max
            print(f"[export] appended code positions: max_position_embeddings {base_max} -> {new_max}")
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    pause_sha256 = None
    if arm == "compressed":
        expected = pause.to(model.get_input_embeddings().weight.dtype).cpu()
        actual = saved_embedding_rows(output, code_ids)
        if not torch.equal(actual, expected):
            max_abs = (actual.float() - expected.float()).abs().max().item()
            raise RuntimeError(
                "saved code embeddings do not match checkpoint pause rows; "
                f"max_abs={max_abs:.6g}"
            )
        pause_sha256 = tensor_sha256(expected)
        print(
            f"[export] verified saved code embeddings sha256={pause_sha256}"
        )
    (output / "compresslm_config.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    manifest = {
        **checkpoint_identity(ckpt_path),
        "arm": arm,
        "base_model": base_model,
        "pause_sha256": pause_sha256,
    }
    (output / EXPORT_MANIFEST).write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"[export] wrote {output}")


if __name__ == "__main__":
    main()
