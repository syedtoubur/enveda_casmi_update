"""c3 signal audit (Kaggle CPU). Inputs: the Stage A A3 tables (kernel_sources), the kit (truths, dumps for the leak flag),
val.parquet + pseudo_spectra.pkl (spectra for parent nl), stage_a_frozen.json. Pre-registered: gate_report 2026-10-18
"c3 signal audit". pseudo_c3 half B enters no verdict (diagnostics only)."""
import glob as _g, sys as _s
_s.path.insert(0, __import__("os").path.dirname(sorted(_g.glob("/kaggle/input/**/code/kit_setup.py", recursive=True), key=len)[0]))
import csv, glob, json, os, pickle, time
import kit_setup as ks
ks.pip_rdkit(); CD = ks.code_dir(); R = ks.kit_root()
from pathlib import Path
import numpy as np, pandas as pd
import harness, a4, audit, iso_nl
from common import split_of
OUT = "/kaggle/working"
cp = harness.load_module(Path(CD) / "casmi_post.py", "casmi_post_audit")
T = harness.truths()
COH = ["C1_npx", "C1_xlib", "C2", "C2_hard"]
INF = float("inf")
log = open(f"{OUT}/audit.log", "w")
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()

def load(coh):
    t = {}
    for f in sorted(glob.glob(f"/kaggle/input/**/a3_{coh}_s*.pkl", recursive=True)):
        t.update(pickle.load(open(f, "rb")))
    return t
ps = pickle.load(open(ks.one("/kaggle/input/**/a3_pseudo_c3.pkl"), "rb"))
coh_t = {c: load(c) for c in COH}
km = {}
for t in [ps] + list(coh_t.values()):
    for r in t.values(): km.update(r["keymap"])
a4.patch_keys(km)
fz = json.load(open(ks.one("/kaggle/input/**/stage_a_frozen.json")))
frag_lists = tuple(l for l in fz["lists"] if l not in ("ice", "gl")) or ("frag",)
assert set(frag_lists) == set(audit.CPU_LISTS), frag_lists
fcfg = dict(ISO_M_ICE=0.0, ISO_LISTS=frag_lists)
frozen = dict(ISO_M_ICE=fz["m_ice"], ISO_LISTS=tuple(fz["lists"]))
def tt(m): return cp.canonicalize_smiles_to_ik14(T[m]["smiles"])

val = pd.read_parquet(ks.one("/kaggle/input/**/val.parquet"))
vg = {m: g for m, g in val.groupby("molecule_id")}
def spectra_val(m):                                   # exactly a3_cohorts_kernel.spectra_for
    g = vg.get(m)
    if g is None: return []
    return [dict(mz=np.asarray(r.ms2_mzs, np.float64), it=np.asarray(r.ms2_normalized_intensities, np.float64),
                 adduct=r.adduct, prec=float(r.precursor_mz), mode=1 if r.ionization_mode == "positive" else -1)
            for r in g.itertuples()][:16]
pspec = pickle.load(open(ks.one("/kaggle/input/**/pseudo_spectra.pkl"), "rb"))["spectra"]

def parent_nl(r, spectra):
    if not spectra: return {}
    v = iso_nl.nl_scores(spectra, [p["smiles"] for p in r["parents"]])
    return {p["ik14"]: (None if not np.isfinite(v[i]) else float(v[i])) for i, p in enumerate(r["parents"]) if p["ik14"]}

rows, res = [], dict(prereg="gate_report 2026-10-18 c3 signal audit", cpu_lists=list(audit.CPU_LISTS), frozen=fz)

# ---- C1/C2: signals over every molecule, leak flag from the dump's BASE #1
c12_sigs, c12_other = [], []
res["c12"] = {}
for coh, t in coh_t.items():
    state, deps, _ = harness.load_dump(coh, Path(CD) / "casmi_post.py")
    base1 = {m: (state["base"][m][0][0] if state["base"].get(m) and state["base"][m][0] else None) for m in t}
    del state, deps
    sigs, other, leaks, t0 = [], [], {}, time.time()
    for i, (m, r) in enumerate(sorted(t.items())):
        truth = tt(m)
        s = audit.signals(r, parent_nl(r, spectra_val(m)), truth)
        leak = bool(base1[m]) and cp.canonicalize_smiles_to_ik14(base1[m]) == truth
        half = split_of(T[m]["smiles"], m)
        sigs.append(s); leaks[m] = leak
        if not leak: other.append(s)
        rows.append(dict(cohort=coh, mid=m, half=half, leak=int(leak), **{k: s[k] for k in ("has", "pos", "mult", "dnl", "dfrag", "nl_c", "nl_p", "hit")},
                         **{f"fire_{k}": int(v) for k, v in audit.fires(s).items()}))
        if (i + 1) % 100 == 0: P(coh, i + 1, f"{time.time() - t0:.0f}s")
    mids = sorted(t)
    truth = {m: tt(m) for m in mids}
    hb = [m for m in mids if split_of(T[m]["smiles"], m) == "B"]
    san = [a4.replay_one(t[m], truth[m], fcfg, frag_only=True) for m in hb]
    tail = {m: a4.replay_one(t[m], truth[m], dict(ISO_M_ICE=INF, ISO_LISTS=frag_lists), frag_only=True) for m in mids}
    oth = [m for m in mids if not leaks[m]]
    td = lambda ms: float(np.mean([tail[m]["rr1"] - tail[m]["rr0"] for m in ms])) if ms else None
    res["c12"][coh] = dict(all=audit.summarize(sigs), other=audit.summarize(other),
                           sanity_frag_only_fire_B=float(np.mean([x["fired"] for x in san])) if san else None,
                           tail_only_delta_all=td(mids), tail_only_delta_other=td(oth),
                           tail_only_down=int(sum(tail[m]["rr1"] < tail[m]["rr0"] for m in mids)))
    c12_sigs += sigs; c12_other += other
    P(coh, "fire", {k: round(v["fire"], 4) for k, v in res["c12"][coh]["all"]["rules"].items()},
      "sanity_B", res["c12"][coh]["sanity_frag_only_fire_B"])
res["c12_pooled"] = audit.summarize(c12_sigs)
res["c12_pooled_other"] = audit.summarize(c12_other)

# ---- pseudo_c3: half A decides TPR; half B diagnostics only
side = {m: split_of(T[m]["smiles"], m) for m in ps}
res["pseudo_c3"] = {}
for half in ("A", "B"):
    mids = sorted(m for m in ps if side[m] == half)
    truth = {m: tt(m) for m in mids}
    sigs = []
    for m in mids:
        s = audit.signals(ps[m], parent_nl(ps[m], pspec.get(m, [])), truth[m]); sigs.append(s)
        rows.append(dict(cohort="pseudo_c3", mid=m, half=half, leak=0, **{k: s[k] for k in ("has", "pos", "mult", "dnl", "dfrag", "nl_c", "nl_p", "hit")},
                         **{f"fire_{k}": int(v) for k, v in audit.fires(s).items()}))
    ft = [audit.fired_truth(ps[m], truth[m], frozen) for m in mids]
    tail = [a4.replay_one(ps[m], truth[m], dict(frozen, ISO_M_ICE=INF)) for m in mids]
    res["pseudo_c3"][half] = dict(summary=audit.summarize(sigs), frozen_fired=sum(f for f, _ in ft),
                                  frozen_fired_truth=sum(h for _, h in ft),
                                  tail_only_delta=float(np.mean([x["rr1"] - x["rr0"] for x in tail])),
                                  role="decides TPR" if half == "A" else "diagnostic only (spent)")
    P("pseudo_c3", half, "h_top", res["pseudo_c3"][half]["summary"]["h_top"], "frozen fired/truth",
      res["pseudo_c3"][half]["frozen_fired"], res["pseudo_c3"][half]["frozen_fired_truth"])

# ---- verdicts
ref = dict(C1_npx=0.84, C1_xlib=0.75, C2=0.64, C2_hard=0.67)
san_ok = all(abs(res["c12"][c]["sanity_frag_only_fire_B"] - v) <= 0.005 for c, v in ref.items())
res["sanity"] = dict(ok=san_ok, ref=ref, got={c: res["c12"][c]["sanity_frag_only_fire_B"] for c in ref})
res["verdicts"] = audit.verdicts(res["c12_pooled"], res["pseudo_c3"]["A"]["summary"])
res["all_dead"] = not any(v["alive"] for v in res["verdicts"].values())
res["status"] = "VOID (sanity failed)" if not san_ok else ("ALL DEAD" if res["all_dead"] else "SOME ALIVE")
json.dump(res, open(f"{OUT}/audit_result.json", "w"), indent=1, default=str)
with open(f"{OUT}/audit_rows.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
P("SANITY", res["sanity"]); P("VERDICTS", json.dumps(res["verdicts"])); P("STATUS", res["status"])
