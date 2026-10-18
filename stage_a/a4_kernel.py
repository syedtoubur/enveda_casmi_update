"""Stage A4/A5 (Kaggle CPU). MODE=freeze: half-A only -> stage_a_frozen.json. MODE=score: frozen choices on half B.
Inputs: A3 outputs (kernel_sources) a3_*.pkl; the kit (truths); the code dataset (stage_a_frozen.json for MODE=score)."""
import glob as _g, sys as _s
_s.path.insert(0, __import__("os").path.dirname(sorted(_g.glob("/kaggle/input/**/code/kit_setup.py", recursive=True), key=len)[0]))
import glob, json, os, pickle, sys, hashlib
MODE = "__MODE__"
import kit_setup as ks
ks.pip_rdkit(); CD = ks.code_dir(); R = ks.kit_root()
from pathlib import Path
import numpy as np
import harness, a4
from common import split_of, scaffold_smiles
OUT = "/kaggle/working"
cp = harness.load_module(Path(CD) / "casmi_post.py", "casmi_post_a4")
T = harness.truths()
COH = ["C1_npx", "C1_xlib", "C2", "C2_hard"]

def load(coh):
    t = {}
    for f in sorted(glob.glob(f"/kaggle/input/**/a3_{coh}_s*.pkl", recursive=True)):
        t.update(pickle.load(open(f, "rb")))
    return t
ps = pickle.load(open(ks.one("/kaggle/input/**/a3_pseudo_c3.pkl"), "rb"))
km = {}
for r in ps.values(): km.update(r["keymap"])
coh_t = {c: load(c) for c in COH}
for t in coh_t.values():
    for r in t.values(): km.update(r["keymap"])
a4.patch_keys(km)
def tt(m): return cp.canonicalize_smiles_to_ik14(T[m]["smiles"])
side = {m: split_of(T[m]["smiles"], m) for m in ps}                # grouping only
mids_a = [m for m in ps if side[m] == "A"]; mids_b = [m for m in ps if side[m] == "B"]
print("pseudo_c3 split A/B", len(mids_a), len(mids_b), flush=True)

if MODE == "freeze":
    truth_a = {m: tt(m) for m in mids_a}                            # half-B truths are never canonicalised here
    fz = a4.freeze(ps, truth_a, mids_a)
    fz.update(split_sizes=dict(A=len(mids_a), B=len(mids_b)))
    s = json.dumps(fz, indent=1, sort_keys=True)
    open(f"{OUT}/stage_a_frozen.json", "w").write(s)
    print(s, "\nsha256", hashlib.sha256(s.encode()).hexdigest())
else:
    fz = json.load(open(ks.one("/kaggle/input/**/stage_a_frozen.json")))
    from rdkit import Chem
    def tags(m):
        mol = Chem.MolFromSmiles(T[m]["smiles"])
        return dict(acyclic=mol.GetRingInfo().NumRings() == 0)
    cfg = dict(ISO_M_ICE=fz["m_ice"], ISO_LISTS=tuple(fz["lists"]))
    fast = dict(ISO_M_ICE=0.0, ISO_LISTS=a4.CAND_LISTS)
    res = dict(frozen=fz)
    def run_c3(c, mids):
        truth = {m: tt(m) for m in mids}
        d = [a4.replay_one(ps[m], truth[m], c) for m in mids]
        grp = [scaffold_smiles(T[m]["smiles"]) or f"acyc:{m}" for m in mids]
        tg = [tags(m) for m in mids]
        st = dict(acyclic=[x["acyclic"] for x in tg], ring=[not x["acyclic"] for x in tg],
                  timstof=[bool(ps[m]["timstof"]) for m in mids], not_timstof=[not ps[m]["timstof"] for m in mids])
        s = a4.summarize(d, grp, st)
        # tail vs fire attribution
        d2 = [a4.replay_one(ps[m], truth[m], dict(c, ISO_TAIL=False)) for m in mids]
        s["delta_fire_only"] = float(np.mean([x["rr1"] - x["rr0"] for x in d2]))
        return s
    res["pseudo_c3_B"] = run_c3(cfg, mids_b)
    res["pseudo_c3_A_ref"] = run_c3(cfg, mids_a)
    res["fast_ship_B"] = run_c3(fast, mids_b)
    res["fast_ship_final_row_parents_B"] = dict(note="identical by construction: parents = top 10 of cp.run's V22 output "
        "(the final row) in both Stage A and the fast-ship; supplementary, not used for any choice", **res["fast_ship_B"])
    res["truth_in_MSG"] = "not measurable: msg_ik14.npy is only on the Colab Drive"
    frag_lists = tuple(l for l in fz["lists"] if l not in ("ice", "gl")) or ("frag",)
    fcfg = dict(ISO_M_ICE=0.0, ISO_LISTS=frag_lists)
    fast_f = dict(ISO_M_ICE=0.0, ISO_LISTS=tuple(l for l in a4.CAND_LISTS if l != "ice"))
    res["c12"] = {}
    for coh, t in coh_t.items():
        mids = list(t)
        out = {}
        for half in ("B", "all"):
            sel = [m for m in mids if half == "all" or split_of(T[m]["smiles"], m) == half]
            truth = {m: tt(m) for m in sel}
            for nm, c in (("frozen_frag_only", fcfg), ("fast_ship_frag_only", fast_f)):
                d = [a4.replay_one(t[m], truth[m], c, frag_only=True) for m in sel]
                grp = [scaffold_smiles(T[m]["smiles"]) or f"acyc:{m}" for m in sel]
                s = a4.summarize(d, grp)
                d2 = [a4.replay_one(t[m], truth[m], dict(c, ISO_TAIL=False), frag_only=True) for m in sel]
                s["delta_wc_fire_only"] = float(np.mean([x["rr_wc"] - x["rr0"] for x in d2]))
                out[f"{half}/{nm}"] = s
        res["c12"][coh] = out
    def comp(nm, key):
        g = lambda coh: res["c12"][coh][f"B/{nm}"][key]
        c1 = (g("C1_npx") + g("C1_xlib")) / 2; c2 = (g("C2") + g("C2_hard")) / 2
        return c1, c2
    res["composite"] = {}
    for nm, c3 in (("frozen_frag_only", res["pseudo_c3_B"]["delta"]), ("fast_ship_frag_only", res["fast_ship_B"]["delta"])):
        lo = comp(nm, "delta_wc"); hi = comp(nm, "delta")
        res["composite"][nm] = {str(pi): dict(worst_case=a4.composite(lo[0], lo[1], c3, pi), actual=a4.composite(hi[0], hi[1], c3, pi))
                                for pi in (0.15, 0.25, 0.35)}
    d = res["pseudo_c3_B"]["delta"]
    res["verdict"] = "DEAD" if d < 0.012 else "NOT DEAD"
    res["verdict_rule"] = "DEAD iff Delta_pseudo_c3 (half B) < 0.012"
    json.dump(res, open(f"{OUT}/stage_a_result.json", "w"), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))
