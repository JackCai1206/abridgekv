#!/usr/bin/env python3
"""Build the extended competition-math eval suite as data/math_eval/<set>.jsonl ({problem_idx, problem, answer}, the format the
MATH-500/HMMT path consumes) plus data/math_eval/SUITE.json (set -> source, n, era). Needs Hub access. AIME 2024 (I, II) and
AIME 2025 come from scripts/dl_math_eval.py. OUT=<dir> overrides the output directory.

  era "post_cutoff": competitions from 2025 on: aime_2025, hmmt_feb_2025, brumo_2025 (30), cmimc_2025 (40), smt_2025 (53),
      aime_2026 (30)
  era "pre_2025": older sets (may overlap public training corpora; the same for every arm): aime_2022 (30, AI-MO),
      aime_2023 (30), aime_2024, hmmt_feb_2023 (30), hmmt_feb_2024 (30), amc_2022_2023 (83 AMC-12, AI-MO)
  built but not in the suite (too hard for a 4B SFT model to separate arms): apex_2025, beyond_aime
Answers are normalized to the competition's canonical string (AMC floats "142.0" -> "142", AIME "033" -> "33").
"""
import json, re, sys
from pathlib import Path
from datasets import load_dataset

import os
OUT = Path(os.environ.get("OUT", Path(__file__).resolve().parents[1] / "data" / "math_eval"))
OUT.mkdir(parents=True, exist_ok=True)


def norm_answer(a) -> str:
    s = str(a).strip()
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    if re.fullmatch(r"0\d+", s):
        s = str(int(s))
    return s


def write(name, rows, source, era):
    path = OUT / f"{name}.jsonl"
    with path.open("w") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({"problem_idx": i, "problem": r["problem"], "answer": norm_answer(r["answer"]),
                                **({"problem_type": r["problem_type"]} if r.get("problem_type") else {})}) + "\n")
    print(f"[suite] {name:16s} n={len(rows):3d} era={era:12s} <- {source}")
    return {"source": source, "n": len(rows), "era": era, "file": f"data/math_eval/{name}.jsonl"}


def matharena(name):
    return [dict(r) for r in load_dataset(f"MathArena/{name}", split="train")]


manifest = {}
for name in ["brumo_2025", "cmimc_2025", "smt_2025", "apex_2025", "aime_2026"]:
    manifest[name] = write(name, matharena(name), f"MathArena/{name}", "excluded_too_hard" if name == "apex_2025" else "post_cutoff")
manifest["hmmt_feb_2023"] = write("hmmt_feb_2023", matharena("hmmt_feb_2023"), "MathArena/hmmt_feb_2023", "pre_2025")
manifest["aime_2023"] = write("aime_2023", matharena("aime_2023_I") + matharena("aime_2023_II"), "MathArena/aime_2023_I+II", "pre_2025")
aimo_aime = [dict(r) for r in load_dataset("AI-MO/aimo-validation-aime", split="train") if "2022" in r["url"]]
assert len(aimo_aime) == 30, len(aimo_aime)
manifest["aime_2022"] = write("aime_2022", aimo_aime, "AI-MO/aimo-validation-aime (url 2022)", "pre_2025")
amc = [dict(r) for r in load_dataset("AI-MO/aimo-validation-amc", split="train")]
manifest["amc_2022_2023"] = write("amc_2022_2023", amc, "AI-MO/aimo-validation-amc", "pre_2025")
beyond = [dict(r) for r in load_dataset("ByteDance-Seed/BeyondAIME", split="test")]
manifest["beyond_aime"] = write("beyond_aime", beyond, "ByteDance-Seed/BeyondAIME (test)", "excluded_too_hard")
# AIME 2024 and 2025 are written by scripts/dl_math_eval.py
manifest["aime_2024"] = {"source": "MathArena/aime_2024_I+II", "n": 30, "era": "pre_2025", "file": "eval_aime (aime2024)"}
manifest["aime_2025"] = {"source": "MathArena/aime_2025", "n": 30, "era": "post_cutoff", "file": "eval_aime (aime2025)"}
manifest["hmmt_feb_2024"] = write("hmmt_feb_2024", matharena("hmmt_feb_2024"), "MathArena/hmmt_feb_2024", "pre_2025")
manifest["hmmt_feb_2025"] = write("hmmt_feb_2025", matharena("hmmt_feb_2025"), "MathArena/hmmt_feb_2025", "post_cutoff")
json.dump(manifest, open(OUT / "SUITE.json", "w"), indent=1)
print("[suite] wrote", OUT / "SUITE.json", "| post-cutoff problems:", sum(m["n"] for m in manifest.values() if m["era"] == "post_cutoff"),
      "| all:", sum(m["n"] for m in manifest.values()))
