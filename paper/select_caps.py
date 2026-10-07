#!/usr/bin/env python3
"""Choose each arm's concurrency cap on one node and report the OTHER node's cell at that cap (user 2026-09-24).

Picking an arm's fastest cap and reporting that same cell reads slightly optimistic (a minimum over noisy cells). Here the
cap is chosen on SELECT_DIR (the replicate) and the reported number is REPORT_DIR's cell at that cap. Arms measured only
on REPORT_DIR (R-KV) fall back to choosing there. Caps measured only on REPORT_DIR (a neighbour added later) join the
choice with REPORT_DIR's value. Writes a directory of symlinks, one cell per arm, for collect_complete.py (COMP_DIR)."""
import glob, json, os, re, sys
SEL, REP, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
def cells(d):
    r = {}
    for f in glob.glob(os.path.join(d, "*_complete_N512_S1_M*_r1.json")):
        m = re.match(r"(.+)_complete_N512_S1_M(\d+)_r1\.json", os.path.basename(f))
        r.setdefault(m.group(1), {})[int(m.group(2))] = (json.load(open(f))["s_per_problem"], f)
    return r
sel, rep = cells(SEL), cells(REP)
os.makedirs(OUT, exist_ok=True)
for f in glob.glob(os.path.join(OUT, "*.json")): os.remove(f)
for arm in sorted(rep):
    pool = dict(sel.get(arm, {}))
    for cap, v in rep[arm].items(): pool.setdefault(cap, v)          # caps only the report node has
    cap = min(pool, key=lambda c: pool[c][0])
    how = "chosen on %s" % ("select" if cap in sel.get(arm, {}) else "report")
    if cap not in rep[arm]: print("%-15s cap %d has no report cell; skipped" % (arm, cap)); continue
    v, f = rep[arm][cap]
    os.symlink(f, os.path.join(OUT, os.path.basename(f)))
    print("%-15s cap %4d (%s): report %.3f  select %s" % (arm, cap, how, v, "%.3f" % sel[arm][cap][0] if cap in sel.get(arm, {}) else "-"))
