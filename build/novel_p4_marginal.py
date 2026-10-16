"""K1 marginal: union top-1 / top-25 hits on pseudo_c3 molecules whose truth is NOT in Engine 1's BASE list."""
import json, pickle, sys
from pathlib import Path
sys.path.insert(0, "E:/ENVEDA_casmi/research/harness")
import harness

state, _, cp = harness.load_dump("pseudo_c3", post=Path("E:/casmi/lines/c3-commit/build/casmi_post.py"))
key = cp.canonicalize_smiles_to_ik14
T = harness.truths()
nov = pickle.load(open("E:/casmi/c3_gen/novel.pkl", "rb"))["novel"]
mids = sorted(nov)
rows = []
for m in mids:
    tk = key(T[m]["smiles"])
    base = [key(s) for s in state["base"][m][0]] if m in state["base"] else []
    in_base = tk in base
    base_top1 = bool(base) and base[0] == tk
    ranks = [i for i, r in enumerate(nov[m], 1) if r[1] == tk]
    rows.append(dict(m=m, in_base=in_base, base_top1=base_top1, r=ranks[0] if ranks else None))
out = {}
for name, sel in [("all", rows), ("truth_in_BASE", [r for r in rows if r["in_base"]]),
                  ("truth_NOT_in_BASE", [r for r in rows if not r["in_base"]])]:
    n = len(sel)
    out[name] = dict(n=n, top1=sum(r["r"] == 1 for r in sel), top25=sum(r["r"] is not None for r in sel),
                     base_top1=sum(r["base_top1"] for r in sel))
print(json.dumps(out, indent=1))
json.dump(out, open("E:/ENVEDA_casmi/research/exploit/20261015-c3-gen-union/k1_marginal.json", "w"), indent=1)
