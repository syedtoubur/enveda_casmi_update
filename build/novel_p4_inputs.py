"""Local CPU step of c3-gen-union (K1): pseudo_c3 dump -> the Colab upload `p4_inputs.pkl`.

Per pseudo_c3 molecule, from the E006 replay of the pool-banned dump (no oracle, no truth):
  * `formulas`  - predicted formulas: distinct formulas of the E006 fused list in rank order, then Engine 2 and PubChem
                  lists, first FORMULA_K (`novel_p4.predicted_formulas`);
  * `mutants`   - c3v2 signed edits (<= 2 moves, +-3 mDa) of the top ANCHORS shifted spectral analogs that Engine 2
                  exported from its BANNED pool (`eng_lists.json.c3.json`, patch_harness_dump ENG_C3_NEW), NO f.z
                  scorer (gate amendment 2026-10-15), capped per molecule; each with its own formula for teacher
                  forcing. Fused-list structures cannot anchor: they sit at the query mass, and delta 0 is no edit;
  * diagnostics - `has_formula` (truth formula among the predicted; read here only to REPORT the ceiling, never used
                  to choose anything) and `acyclic`.

    python build/novel_p4_inputs.py [--dump pseudo_c3] [--anchors 10] [--cap 300] [--procs 8] [--limit 0]
"""
from __future__ import annotations

import argparse
import collections
import os
import pickle
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "E:/ENVEDA_casmi/research/harness")

import novel_decoder as nd  # noqa: E402
import novel_p4 as p4  # noqa: E402

E006 = dict(METRIC_FUSION=True, ICE_FALLBACK=True, FUSION_WINDOW=40)
OUT = "E:/casmi/c3_gen/p4_inputs.pkl"
_R = {}


def _formula(smi):
    Chem = nd._chem()
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if m is None:
        return None
    c = nd.mol_formula(m)
    return "".join(f"{e}{c[e] if c[e] > 1 else ''}" for e in sorted(c, key=lambda e: (e != "C", e != "H", e)))


def _init():
    import delta_m_mutator as dm
    _R["dm"] = dm
    reach = dm.build_reach_signed(maxops=2)
    _R["reach"], _R["idx"] = reach, dm.mass_index(reach)


def _mutants(task):
    """task = (mid, query neutral mass, [(anchor smiles, anchor exact mass, similarity)], n anchors, cap)."""
    mid, target, ana, n_anchors, cap = task
    dm = _R["dm"]
    st = collections.Counter()
    pairs = []
    for smi, mass, sim in sorted(ana, key=lambda a: -a[2]):
        dv = [d for d in dm.match_mass_delta(target - float(mass), _R["idx"], 0.003)
              if min(len(x) for x in _R["reach"][d]) <= 2]
        if dv:
            pairs.append((smi, dv))
        if len(pairs) >= n_anchors:
            break
    t0 = time.monotonic()
    got = dm.mutate_anchored(pairs, _R["reach"], cap=cap, max_steps=2, deadline=t0 + 60, stats=st)
    out = []
    for s, ai, dec in got:
        f = _formula(s)
        if f:
            out.append((s, f, ai, "+".join(dec)))
    return mid, out, dict(st), round(time.monotonic() - t0, 2)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="pseudo_c3")
    ap.add_argument("--anchors", type=int, default=5, help="MUT_ANCHORS of the c3v2 runner")
    ap.add_argument("--c3json", default=None, help="Engine 2 analog export; default dumps/<dump>/eng_lists.json.c3.json")
    ap.add_argument("--cap", type=int, default=300)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cohort", default="pseudo_c3", help="truth cohort to keep (C2 etc. only for a plumbing smoke)")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    import harness
    state, deps, cp = harness.load_dump(a.dump, __import__("pathlib").Path(HERE) / "casmi_post.py")
    T = harness.truths()
    mids = [m for m in state["te_mids"] if m in T and T[m].get("cohort") == a.cohort]
    if a.limit:
        mids = mids[:a.limit]
    fused = cp.run(state, deps, **E006)
    import json
    import pandas as pd
    c3j = json.load(open(a.c3json or os.path.join(str(harness.DUMPS), a.dump, "eng_lists.json.c3.json")))
    tims = dict(pd.read_parquet(harness.PSEUDO_C3, columns=["molecule_id", "timstof"]).itertuples(index=False))
    Chem = nd._chem()
    rows, tasks = {}, []
    for m in mids:
        lst = list(fused.get(m, []))
        eng = list(((state.get("eng") or {}).get(m) or {}).get("smiles") or [])
        pc = list(((state.get("pc") or {}).get(m) or {}).get("pc") or [])
        ranked = lst + [s for s in eng + pc if s not in set(lst)]
        forms = [_formula(s) for s in ranked[:60]]
        pred = p4.predicted_formulas(forms, k=p4.FORMULA_K, allowed=nd.ALLOWED_ELEMENTS)
        tm = Chem.MolFromSmiles(T[m]["smiles"])
        truth_f = _formula(T[m]["smiles"])
        rows[m] = dict(molecule_id=m, formulas=pred, has_formula=truth_f in pred, truth_formula=truth_f,
                       acyclic=tm is not None and tm.GetRingInfo().NumRings() == 0,
                       target=float(state["targets"][m]), fused_top=lst[:25])
        tgt = float(state["targets"][m]) * (1.0 - (p4.PPM_CENTER * 1e-6 if tims.get(m) else 0.0))
        rows[m]["timstof"] = bool(tims.get(m))
        tasks.append((m, tgt, (c3j.get(m) or {}).get("ana") or [], a.anchors, a.cap))
    t0 = time.time()
    with Pool(a.procs, _init) as pool:
        for i, (m, muts, st, secs) in enumerate(pool.imap_unordered(_mutants, tasks), 1):
            rows[m].update(mutants=muts, mut_stats=st, mut_secs=secs)
            if i % 50 == 0 or i == len(tasks):
                print(f"  mutants {i}/{len(tasks)} {time.time() - t0:.0f}s", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    pickle.dump(dict(rows=rows, cfg=dict(E006=E006, anchors=a.anchors, cap=a.cap, formula_k=p4.FORMULA_K)),
                open(a.out, "wb"), protocol=4)
    n = len(rows)
    print(f"wrote {a.out}: {n} molecules | formula recall (ceiling, diagnostic) "
          f"{sum(r['has_formula'] for r in rows.values()) / max(n, 1):.3f} | "
          f"molecules with mutants {sum(bool(r['mutants']) for r in rows.values())} | "
          f"mean mutants {sum(len(r['mutants']) for r in rows.values()) / max(n, 1):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
