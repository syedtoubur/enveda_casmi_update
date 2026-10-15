"""K1 verdict for c3-gen-union, locally, from the Colab `p4_raw.pkl` + `p4_inputs.pkl` (campaign 20261015-c3-gen-slot2).

Ranks per molecule with the metric's tautomer-canonical IK14 (`casmi_post.canonicalize_smiles_to_ik14`):
  arms   dec (beam 64, predicted formulas) | mut (teacher-scored edits) | union = dec + mut   [K1 is read on `union`]
         dec256 / union256 on the beam-256 subset, beside dec / union on the SAME subset (what beam 256 would add)
Writes  <idea>/k1_result.json + k1_result.md, and the replay sidecar `E:/casmi/c3_gen/novel.pkl`
        = {molecule_id: [(smiles, ik14, score, src)] top NOVEL_KEEP of the union} for c3-detector / c3-slot2-route.

    python build/novel_p4_score.py [--raw E:/casmi/c3_gen/p4_raw.pkl] [--inputs E:/casmi/c3_gen/p4_inputs.pkl]
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "E:/ENVEDA_casmi/research/harness")

import casmi_post as cp  # noqa: E402
import novel_p4 as p4  # noqa: E402

IDEA = "E:/ENVEDA_casmi/research/exploit/20261015-c3-gen-union"
NOVEL_KEEP = 25


def arm_lists(rec, arm):
    if arm == "union":
        return [("dec", [(s, lp) for s, lp, _f in rec["dec"]]), ("mut", [(s, lp) for s, lp, *_ in rec["mut"]])]
    if arm == "union256":
        return [("dec256", [(s, lp) for s, lp, _f in rec["dec256"]]), ("mut", [(s, lp) for s, lp, *_ in rec["mut"]])]
    if arm == "mut":
        return [("mut", [(s, lp) for s, lp, *_ in rec["mut"]])]
    return [(arm, [(s, lp) for s, lp, _f in rec[arm]])]


def score(raw, inputs, truths, arms, mids):
    key = cp.canonicalize_smiles_to_ik14
    rep, ranked_union = {}, {}
    for arm in arms:
        rows = []
        for m in mids:
            ranked = p4.union_rank(arm_lists(raw[m], arm), key)
            tk = key(truths[m]["smiles"])
            r = p4.rank_of(ranked, tk)
            inp = inputs[m]
            rows.append(dict(rank=r, acyclic=inp["acyclic"], timstof=inp.get("timstof", False),
                             has_formula=inp["has_formula"], src_top1=ranked[0][3] if ranked and r == 1 else ""))
            if arm == "union":
                ranked_union[m] = ranked[:NOVEL_KEEP]
        rep[arm] = p4.k1_report(rows)
    return rep, ranked_union


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="E:/casmi/c3_gen/p4_raw.pkl")
    ap.add_argument("--inputs", default="E:/casmi/c3_gen/p4_inputs.pkl")
    ap.add_argument("--sidecar", default="E:/casmi/c3_gen/novel.pkl")
    ap.add_argument("--idea", default=IDEA)
    a = ap.parse_args(argv)
    import harness
    T = harness.truths()
    R = pickle.load(open(a.raw, "rb"))
    raw, sub = R["raw"], set(R.get("sub256") or [])
    inputs = pickle.load(open(a.inputs, "rb"))["rows"]
    mids = sorted(m for m in raw if m in inputs and m in T and not raw[m].get("error"))
    full, ranked = score(raw, inputs, T, ("dec", "mut", "union"), mids)
    subm = [m for m in mids if m in sub and raw[m]["dec256"]]
    on_sub, _ = score(raw, inputs, T, ("dec", "union", "dec256", "union256"), subm) if subm else ({}, {})
    errors = sorted(m for m in raw if raw[m].get("error"))
    res = dict(n=len(mids), errors=errors, beam=R.get("beam"), K1=full["union"]["K1"], h=full["union"]["all"]["h"],
               full=full, beam256_subset=dict(n=len(subm), **on_sub))
    os.makedirs(a.idea, exist_ok=True)
    json.dump(res, open(os.path.join(a.idea, "k1_result.json"), "w"), indent=1)
    pickle.dump(dict(novel=ranked, src="novel_p4_score.py union (decoder beam %s + mutator), leak-free scale"
                     % R.get("beam")), open(a.sidecar, "wb"), protocol=4)
    lines = [f"# K1 result (c3-gen-union), {len(mids)} pseudo_c3 molecules, beam {R.get('beam')}", "",
             f"**K1 {res['K1']}**: union top-1 exact h = {res['h']:.4f} (bar 0.03, gate_report 2026-10-15)", "",
             "| arm | n | h (top-1) | hit@2 | hit@3 | hit@4 | hit@25 | MRR@25 |", "|---|---|---|---|---|---|---|---|"]
    for name, rep in [(f"{k} (all)", v) for k, v in full.items()] + [(f"{k} (beam-256 subset)", v) for k, v in on_sub.items()]:
        b = rep["all"]
        lines.append(f"| {name} | {b['n']} | {b['h']:.4f} | {b['hit@2']:.4f} | {b['hit@3']:.4f} | {b['hit@4']:.4f} | "
                     f"{b['hit@25']:.4f} | {b['mrr25']:.4f} |")
    u = full["union"]
    lines += ["", f"- formula recall (truth formula among the predicted; ceiling for `dec`): {u['formula_recall']:.3f}",
              f"- given the formula: h {u['given_formula']['h']:.4f} (n {u['given_formula']['n']})",
              f"- acyclic h {u['acyclic']['h']:.4f} (n {u['acyclic']['n']}) | ring-bearing h {u['ring']['h']:.4f} "
              f"(n {u['ring']['n']}) | timsTOF h {u['timstof']['h']} (n {u['timstof']['n']})",
              f"- source of top-1 hits: {u['src_of_top1_hits']}", f"- errors: {len(errors)}"]
    open(os.path.join(a.idea, "k1_result.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if res["K1"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
