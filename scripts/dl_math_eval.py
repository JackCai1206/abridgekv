"""Download MATH-500, AIME 2024 (I, II) and AIME 2025 into data/math_eval/ (OUT=<dir> overrides). Needs Hub access."""
import os, json
for k in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE"): os.environ.pop(k, None)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
from datasets import load_dataset

OUT = os.environ.get("OUT", os.path.join(REPO, "data/math_eval"))
os.makedirs(OUT, exist_ok=True)

def dump(name, rows):
    p = os.path.join(OUT, name)
    with open(p, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print(f"  saved {len(rows):4d} -> {p}", flush=True)

# MATH-500
try:
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    dump("math500.jsonl", [{"problem": r["problem"], "answer": r.get("answer"), "solution": r.get("solution"),
                            "subject": r.get("subject"), "level": r.get("level")} for r in ds])
except Exception as e:
    print("MATH-500 FAILED:", e, flush=True)

# AIME 2024 (I+II) and 2025
for tag, spec in [("aime2024_I", "MathArena/aime_2024_I"), ("aime2024_II", "MathArena/aime_2024_II"),
                  ("aime2025", "MathArena/aime_2025")]:
    try:
        ds = load_dataset(spec, split="train")
        dump(f"{tag}.jsonl", [{"problem": r.get("problem") or r.get("question"),
                               "answer": r.get("answer"), "id": r.get("id") or r.get("problem_id")} for r in ds])
    except Exception as e:
        print(f"{tag} ({spec}) FAILED:", e, flush=True)

print("DONE. files:", os.listdir(OUT), flush=True)
