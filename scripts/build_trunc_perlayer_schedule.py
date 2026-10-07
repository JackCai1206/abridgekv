"""Per-layer schedule from the single-layer truncation probe (the S-PL-trunc recipe, docs/SCHEDULES.md): rank layers by
the held-out CE cost of halving their code rows (128 -> 64) on a trained uniform model; the TOP_N costliest get 256 rows,
the BOTTOM_N cheapest 64, the rest keep 128 (mean 128 for 6/12 of 36). Writes the trainer's schedule format
(NL rows x NH heads). Usage: PROBE_JSONL=<jsonl> OUT=<json> [TOP_N=6 BOTTOM_N=12 NH=8] python3 scripts/build_trunc_perlayer_schedule.py"""
import json, os, re, sys
J = os.environ["PROBE_JSONL"]; OUT = os.environ["OUT"]
TOP, BOT, NH = int(os.environ.get("TOP_N", 6)), int(os.environ.get("BOTTOM_N", 12)), int(os.environ.get("NH", 8))
recs = {}
for line in open(J):
    try: r = json.loads(line); recs[r["name"]] = r
    except Exception: pass
base = recs["trained"]["held_out_ce"]
cost = {}
for name, r in recs.items():
    m = re.fullmatch(r"layer(\d+)->64", name)
    if m and r["set"] == "full": cost[int(m.group(1))] = r["held_out_ce"] - base
NL = max(cost) + 1
missing = [l for l in range(NL) if l not in cost]
if missing: sys.exit(f"missing layers {missing}")
order = sorted(cost, key=lambda l: -cost[l])
up, down = set(order[:TOP]), set(order[-BOT:])
sched = [[256 if l in up else 64 if l in down else 128 for _ in range(NH)] for l in range(NL)]
mean = sum(map(sum, sched)) / (NL * NH)
json.dump(sched, open(OUT, "w"))
print(f"trained CE {base:.5f}; all->64 {recs.get('all->64', {}).get('held_out_ce', float('nan')):.5f}")
print("cost of halving, by layer (nats x1e3):", " ".join(f"{l}:{1e3*cost[l]:.2f}" for l in range(NL)))
print(f"256: {sorted(up)}\n 64: {sorted(down)}\nmean {mean:.1f} -> {OUT}")
