"""Prep a CORRECTLY-FORMATTED math SFT set for the compression pilot, fixing the two bugs in the old OpenThoughts
prep: (1) response now ENDS with EOS so termination is trainable; (2) source is nvidia/OpenMathReasoning (CoT +
\\boxed{} answers), and we KEEP only \\boxed{}-terminated solutions so the boxed-answer format is taught. Prompt is
built with the SAME format_user_content(+boxed instruction) the eval uses, so train/eval prompts match.

MULTIPROCESS: the main process streams raw (problem, solution) text; a Pool tokenizes buffered chunks in parallel
(tokenization of ~1M docs is the bottleneck). imap preserves order, so the sequential band-fill logic is unchanged.
"""
import os, sys
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
for k in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "HF_OFFLINE", "TRANSFORMERS_OFFLINE"): os.environ[k] = "0"  # ONLINE
os.environ["TOKENIZERS_PARALLELISM"] = "false"                 # we parallelize via PROCESSES; avoid thread oversubscription
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from prompt_format import format_user_content
from multiprocessing import Pool

TOK = os.environ.get("TOK_DIR", "Qwen/Qwen3-4B-Base")
OUT = os.environ.get("OUT", os.path.join(REPO, "data/math_sft_pairs.pkl"))
LO, HI = int(os.environ.get("LO", 512)), int(os.environ.get("HI", 4096))
N_KEEP = int(os.environ.get("N_KEEP", 12000))
BOXED = bool(int(os.environ.get("BOXED_INSTRUCTION", "1")))
NPROC = int(os.environ.get("NPROC", str(min(os.cpu_count() or 8, 64))))
CHUNK = int(os.environ.get("PREP_CHUNK", "20000"))             # candidates buffered before a parallel tokenize flush
BANDS = []
if os.environ.get("BANDS"):
    for part in os.environ["BANDS"].split(","):
        edge, frac = part.split(":")
        BANDS.append([int(edge), int(round(float(frac) * N_KEEP)), 0])   # [upper_edge, quota, filled]
    assert sorted(b[0] for b in BANDS) == [b[0] for b in BANDS] and BANDS[-1][0] == HI, "BANDS must ascend and end at HI"

# ---- worker: one tokenizer per process ----
_W = {}
def _init_worker(tok_dir, boxed):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _W["tok"] = AutoTokenizer.from_pretrained(tok_dir)
    _W["eos"] = _W["tok"].eos_token_id
    _W["boxed"] = boxed
def _tok_one(item):
    prob, sol, ans = item
    t = _W["tok"]
    pid = t(format_user_content(prob, _W["boxed"]), add_special_tokens=True).input_ids
    rid = t(sol, add_special_tokens=False).input_ids + [_W["eos"]]
    return pid, rid, ans


def main():
    tok = AutoTokenizer.from_pretrained(TOK); EOS = tok.eos_token_id     # parent: warm cache + stats
    print(f"eos={EOS} boxed={BOXED} window=[{LO},{HI}] target={N_KEEP} NPROC={NPROC} chunk={CHUNK} "
          f"bands={[(b[0],b[1]) for b in BANDS]}", flush=True)
    if os.environ.get("PREP_PARQUET_DIR"):      # offline (compute node): the same cot-*.parquet shards, in name order = Hub streaming order
        import glob
        files = sorted(glob.glob(os.path.join(os.environ["PREP_PARQUET_DIR"], "cot-*.parquet")))
        assert files, f"no cot-*.parquet under {os.environ['PREP_PARQUET_DIR']}"
        print(f"[prep] local parquet: {len(files)} shards from {os.environ['PREP_PARQUET_DIR']}", flush=True)
        ds = load_dataset("parquet", data_files=files, split="train", streaming=True)
    else:
        ds = load_dataset("nvidia/OpenMathReasoning", split="cot", streaming=True)
    kept, seen, done = [], 0, False
    pool = Pool(NPROC, initializer=_init_worker, initargs=(TOK, BOXED))
    buf = []

    def flush():
        nonlocal buf, done
        if not buf:
            return
        cs = max(1, len(buf) // (NPROC * 4))
        for pid, rid, ans in pool.imap(_tok_one, buf, chunksize=cs):     # ORDER-PRESERVING -> band logic unchanged
            tot = len(pid) + len(rid)
            if LO <= tot <= HI:
                if BANDS:
                    band = next(b for b in BANDS if tot <= b[0])
                    if band[2] >= band[1]:
                        continue
                    band[2] += 1
                kept.append({"prompt": np.asarray(pid, dtype=np.int32), "response": np.asarray(rid, dtype=np.int32),   # int32 arrays: ~10x less RAM than lists
                             "n_blocks": (tot + 511) // 512, "answer": ans})
            if BANDS:
                if all(b[2] >= b[1] for b in BANDS):
                    done = True; break
            elif len(kept) >= N_KEEP:
                done = True; break
        buf = []

    for ex in ds:
        seen += 1
        sol = ex.get("generated_solution") or ""; prob = ex.get("problem") or ""
        if "\\boxed" not in sol or not prob:
            continue
        if BANDS:                                                        # cheap char pre-skip once short bands fill
            edges = [0] + [b[0] for b in BANDS[:-1]]
            min_lo = min((edges[i] for i, b in enumerate(BANDS) if b[2] < b[1]), default=None)
            if min_lo and (len(prob) + len(sol)) / 3.5 < 0.5 * min_lo:
                continue
        buf.append((prob, sol, ex.get("expected_answer")))
        if len(buf) >= CHUNK:
            flush()
            bs = " ".join(f"<={b[0]}:{b[2]}/{b[1]}" for b in BANDS) if BANDS else ""
            print(f"  seen {seen} kept {len(kept)} {bs}", flush=True)
            if done:
                break
    if not done:
        flush()
    pool.close(); pool.join()

    HOLD = int(os.environ.get("HOLDOUT_PER_BAND", "0"))
    if BANDS and HOLD:
        def band_of(k):
            t = len(k["prompt"]) + len(k["response"])
            return next(i for i, b in enumerate(BANDS) if t <= b[0])
        front, rest, cnt = [], [], [0] * len(BANDS)
        for k in kept:
            i = band_of(k)
            if cnt[i] < HOLD:
                front.append(k); cnt[i] += 1
            else:
                rest.append(k)
        kept = front + rest
        print(f"holdout: first {len(front)} band-stratified {cnt} -> set EVALN={len(front)}", flush=True)

    torch.save(kept, OUT)
    from collections import Counter
    nb = Counter(k["n_blocks"] for k in kept)
    ends_eos = sum(1 for k in kept if k["response"][-1] == EOS)
    print(f"SAVED {len(kept)} -> {OUT} | ends_with_EOS={ends_eos}/{len(kept)} | n_blocks={dict(sorted(nb.items()))}", flush=True)
    print("median tot tokens:", sorted(len(k['prompt']) + len(k['response']) for k in kept)[len(kept) // 2], flush=True)


if __name__ == "__main__":
    main()
