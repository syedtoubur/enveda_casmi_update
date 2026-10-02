"""Delta-M mutator channel for the v4n fusion notebook (specs_v4n_update M2/M3).

Runs in its own process so its `casmi` snapshot (S1 / casmi26-v3-models, the one that produced the
f.z logits) never collides with the v4n engine's. CPU only: no torch, no GPU, no network.

Per gated molecule:
  1. harvest Murcko scaffolds and target formulas from that molecule's PubChem top-25 and, when
     MUT_ENG points at the engine-A lists, from its engine-A top-40 as well.  Engine A draws on a
     different structure universe (COCONUT + ChEBI/LIPID MAPS + training structures) and ranks it
     by spectral evidence rather than by f.z, so its scaffolds are an orthogonal source;
  2. enumerate decorated structures at one of those exact formulas (delta_m_mutator);
  3. pass 1 -- partial f.z over the ECFP4 block of the selected bits, keep the top MUT_N1;
  4. pass 2 -- full selected-bit fingerprint, full f.z, sort descending;
  5. walk the order, key each structure by tautomer-canonical InChIKey-14, and keep it only if the
     key is absent from the candidate pool (train U COCONUT) and from the molecule's own PubChem
     and engine-A lists -- i.e. only genuinely new structures. Stop at MUT_K.  (The engine's pool
     carries ChEBI/LIPID MAPS, which pool_meta.parquet does not, so the engine keys are not already
     covered by pool_keys; excluding them also stops a candidate from collecting both the engine's
     and the mutator's RRF weight.)

Gate: with MUT_C3 set, `c3_gate.classify` decides per molecule from similarities in [0,1] -- a
strong mass-shifted spectral analog, a weak exact-mass match, and a formula delta realisable in a
couple of transforms.  That gate can legitimately fire on nothing, which the old one could not: it
took the weakest MUT_FRAC of the field by absolute `best_pool_fz` and so always fired on 30% of the
molecules whether or not any was novel, and `best_pool_fz` scores train + COCONUT rather than
PubChem, which is the database Class 3 is defined against.  Without MUT_C3 the old rank gate is
used unchanged, so the PubChem-only and bio notebooks keep their exact behaviour.

Two-stage by necessity: `BASE` (and so `lib_max`) is not built until cell 12 while the mutator is
cell 9, so this stage passes the engine's own exact-mass library similarity for both the Class-1
veto and the exact-match term, and the fusion cell re-runs the gate with v4n's `lib_max` and the
ICEBERG/GLACIER agreement before it promotes anything.

The engine lists arrive through the MUT_ENG environment variable rather than argv, so this
argument contract stays byte-compatible with the PubChem-only notebook.

usage: python mut_runner.py <v3_code_dir> <pool_dir> <pc_lists.json> <pc_z.npz> <out.json>
                            <n_workers> [max_mols]
env:   MUT_ENG=<eng_lists.json>   second scaffold source; unset => PubChem only
       MUT_C3=<c3_diag.json>     engine Class-3 diagnostics; unset => the old MUT_FRAC rank gate
       MUT_K MUT_N1 MUT_FRAC MUT_CAP MUT_DEADLINE
       C3_* thresholds are read by c3_gate itself
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

# the notebook overrides these through the environment so one edit in the config cell is enough
MUT_K = int(os.environ.get("MUT_K", 25))            # candidates kept per molecule
MUT_N1 = int(os.environ.get("MUT_N1", 2000))        # pass-1 survivors scored with the full fingerprint
MUT_FRAC = float(os.environ.get("MUT_FRAC", 0.30))  # share of molecules fired on (weakest best_pool_fz)
MUT_CAP = int(os.environ.get("MUT_CAP", 8000))      # products enumerated per molecule before screening
MUT_ENG = os.environ.get("MUT_ENG", "")             # engine-A lists as a second scaffold source
MUT_DEADLINE = float(os.environ.get("MUT_DEADLINE", 45))   # wall-clock seconds per molecule, generation only
MUT_C3 = os.environ.get("MUT_C3", "")               # engine Class-3 diagnostics; unset => rank gate
MUT_ANA = int(os.environ.get("MUT_ANA", 20))        # analog structures used as a scaffold source

_W = {}


def init_worker(v3code, pool_dir, mut_dir):
    for p in (v3code, mut_dir):
        if p and p not in sys.path:
            sys.path.insert(0, p)
    import pandas as pd
    from casmi import chem

    import c3_gate
    import delta_m_mutator as dm

    _W["chem"] = chem
    _W["dm"] = dm
    _W["c3"] = c3_gate
    bits = np.load(os.path.join(pool_dir, "fp_bits.npy"))
    _W["bits"] = bits
    _W["sel_e"] = np.where(bits < 4096)[0]
    _W["raw_e"] = bits[_W["sel_e"]]
    _W["pool_keys"] = set(pd.read_parquet(os.path.join(pool_dir, "pool_meta.parquet"),
                                          columns=["key"]).key.values.tolist())
    r = chem._rdkit()
    _W["ecfp4"] = r["gen"].GetMorganGenerator(radius=2, fpSize=4096)
    _W["Chem"] = r["Chem"]
    _W["reach"] = dm.build_reach()


def mutate_one(task):
    """task = (mid, z, pc_hits, pc_keys, eng_hits, eng_keys, c3) -> (mid, smiles, f.z, keys, tail).

    `c3` is this molecule's engine diagnostics, or None to skip the Class-3 gate entirely (the
    old rank-gate path, where main() has already chosen which molecules fire).
    """
    mid, z, hits, pc_keys, eng_hits, eng_keys, c3 = task
    t0 = time.time()
    chem, dm, Chem, gen = _W["chem"], _W["dm"], _W["Chem"], _W["ecfp4"]
    diag = dict(molecule_id=mid, n_gen=0, n_pass1=0, n_pass2=0, n_in_pool=0, n_listed=0, tier=None)
    import collections
    st = collections.Counter()

    # ---- Class-3 gate and the analog scaffold source
    ana_smis, reasons = [], collections.Counter()
    if c3 is not None:
        G = _W["c3"]
        ana = c3.get("ana") or []
        exact = float(c3.get("lib_exact") or 0.0)
        ana_max = float(ana[0][2]) if ana else 0.0
        # the target formula hypothesis: the most common formula among the retrieved hits, which is
        # what the generator will aim at.  The analog sits in a +-200 Da window, so the step count
        # must be measured from the analog's scaffold to *this* formula, not to the analog's own.
        tgt = dm.harvest_formulas(list(hits), 25)
        steps = None
        if tgt and ana:
            scafs = dm.harvest_scaffolds([ana[0][0]], 1)
            for _k, sc in scafs.items():
                d = G.delta_of(tgt[0], dm.elem_vec(sc))
                steps = G.min_steps(d, _W["reach"], dm.EL)
                break
        # lib_max is not knowable here (BASE is built after this cell), so the engine's exact-mass
        # library similarity stands in for the Class-1 veto; cell 17 re-checks with the real one.
        tier = G.classify(exact, exact, ana_max, steps, reasons=reasons)
        diag.update(tier=tier, ana_max=round(ana_max, 4), lib_exact=round(exact, 4),
                    delta_steps=(-1 if steps is None else steps),
                    gate=(list(reasons)[0] if reasons else ""))
        if tier is None:
            diag["secs"] = time.time() - t0
            return mid, [], [], [], (diag, [], [])
        ana_smis = [a[0] for a in ana[:MUT_ANA]]

    srcs = [("pc", list(hits))]
    if eng_hits:
        srcs.append(("eng", list(eng_hits)))
    if ana_smis:
        srcs.append(("ana", ana_smis))
    # the analog source donates scaffolds only -- its molecules are at the wrong mass, so letting
    # their formulas become targets would spend the budget on formulas the precursor rules out.
    contrib = [nm != "ana" for nm, _ in srcs]
    caps = {"ana": _W["c3"].MAX_STEPS} if ana_smis else None
    try:
        gen_smis, prov = dm.mutate_sources(srcs, _W["reach"], cap=MUT_CAP, stats=st,
                                           deadline=time.monotonic() + MUT_DEADLINE,
                                           contributes_formulas=contrib, max_steps=caps)
    except Exception as e:                                   # a bad hit must not kill the molecule
        diag.update(error=repr(e)[:200], secs=time.time() - t0)
        return mid, [], [], [], diag
    diag.update(n_gen=len(gen_smis), scaffolds=int(st["scaffolds"]), formulas=int(st["formulas"]),
                pairs_reachable=int(st["pairs_reachable"]), pairs_unreachable=int(st["pairs_unreachable"]),
                n_sc_pc=int(st["sc_pc"]), n_sc_eng=int(st["sc_eng"]),
                n_sc_eng_only=int(st["sc_eng_only"]), n_sc_ana=int(st["sc_ana"]),
                n_sc_ana_only=int(st["sc_ana_only"]), pairs_too_deep=int(st["pairs_too_deep"]),
                hit_deadline=int(st["hit_deadline"]))
    if not gen_smis:
        diag["secs"] = time.time() - t0
        return mid, [], [], [], diag

    # ---- pass 1: ECFP4 block of the selected bits, same screen as the PubChem channel
    raw_e, z = _W["raw_e"], np.asarray(z, np.float32)
    z_e = z[_W["sel_e"]].astype(np.float32)
    n = len(gen_smis)
    E = np.zeros((n, len(raw_e)), np.float32)
    ok = np.zeros(n, bool)
    for i, s in enumerate(gen_smis):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        E[i] = gen.GetFingerprintAsNumPy(m)[raw_e]
        ok[i] = True
    sc1 = E @ z_e
    sc1[~ok] = -np.inf
    n1 = min(MUT_N1, int(ok.sum()))
    diag["n_pass1"] = n1
    if n1 == 0:
        diag["secs"] = time.time() - t0
        return mid, [], [], [], diag
    top1 = np.argpartition(-sc1, n1 - 1)[:n1] if n1 < n else np.where(ok)[0]

    # ---- pass 2: full selected-bit fingerprint, full f.z
    bits = _W["bits"]
    fz, idx = [], []
    for i in top1:
        fp = chem.raw_fingerprint(gen_smis[i])
        if fp is None:
            continue
        fz.append(float(fp[bits].astype(np.float32) @ z))
        idx.append(int(i))
    diag["n_pass2"] = len(idx)
    if not idx:
        diag["secs"] = time.time() - t0
        return mid, [], [], [], diag
    order = np.argsort(-np.asarray(fz), kind="stable")

    # ---- keep only structures new to the pool, to the PubChem list and to the engine list
    known = set(pc_keys) | set(eng_keys) | _W["pool_keys"]
    out, out_fz, out_k, out_f, out_s, seen = [], [], [], [], [], set()
    for o in order:
        s = gen_smis[idx[o]]
        k = chem.score_key(s)
        if k is None or k in seen:
            continue
        seen.add(k)
        if k in known:
            diag["n_in_pool"] += 1
            continue
        out.append(s)
        out_fz.append(float(fz[o]))
        out_k.append(k)
        out_f.append(dm.fmt_vec(dm.vec_of_smiles(s)))
        out_s.append(prov.get(s, "pc"))
        if len(out) >= MUT_K:
            break
    diag["n_listed"] = len(out)
    diag["n_listed_eng"] = sum(1 for x in out_s if x == "eng")
    diag["n_listed_ana"] = sum(1 for x in out_s if x == "ana")
    diag["secs"] = time.time() - t0
    return mid, out, out_fz, out_k, (diag, out_f, out_s)


def main(argv):
    V3CODE, POOL_DIR, PC_JSON, Z_NPZ, OUT, NW = argv[:6]
    MAXM = int(argv[6]) if len(argv) > 6 else 0
    NW = int(NW)
    MUT_DIR = os.path.dirname(os.path.abspath(__file__))
    T0 = time.time()

    PC = json.load(open(PC_JSON))
    zf = np.load(Z_NPZ, allow_pickle=True)
    Z = {str(m): zf["z"][i] for i, m in enumerate(zf["mids"])}
    EN = {}
    if MUT_ENG and os.path.exists(MUT_ENG):
        try:
            EN = json.load(open(MUT_ENG))
        except Exception as e:                   # a missing or broken second source must not stop us
            print("eng lists unreadable, PubChem only:", repr(e), flush=True)
    C3 = {}
    if MUT_C3 and os.path.exists(MUT_C3):
        try:
            C3 = json.load(open(MUT_C3))
        except Exception as e:               # no diagnostics -> the old rank gate, never a crash
            print("c3 diagnostics unreadable, falling back to the rank gate:", repr(e), flush=True)
    print(f"pc lists {len(PC)}, z vectors {len(Z)}, eng lists {len(EN)}, c3 diagnostics {len(C3)}"
          + ("" if EN else "  (PubChem only)"), flush=True)
    if C3:
        # The gate thresholds are provisional until GC.3, and a threshold set too high fires on
        # nothing and wastes a submission silently.  Print the distributions the gate reads so one
        # run is enough to calibrate them, whatever the scale turns out to be.
        import c3_gate as _G
        _am = np.array([float((v.get("ana") or [[0, 0, 0.0]])[0][2]) for v in C3.values()])
        _lx = np.array([float(v.get("lib_exact") or 0.0) for v in C3.values()])
        _pc = [1, 10, 25, 50, 75, 90, 99]
        print("C3 SIGNALS (percentiles " + "/".join(str(x) for x in _pc) + "):", flush=True)
        for nm, arr in (("ana_max", _am), ("lib_exact", _lx), ("margin", _am - _lx)):
            print("  %-10s " % nm + " ".join("%6.3f" % x for x in np.percentile(arr, _pc)),
                  flush=True)
        print("  thresholds: ana_max>=%.2f margin>=%.2f lib_max<%.2f steps<=%d"
              % (_G.ANA_MIN, _G.MARGIN_MIN, _G.LIB_TAU, _G.MAX_STEPS), flush=True)
        print("  would pass ana_max alone: %d of %d | and the margin too: %d"
              % (int((_am >= _G.ANA_MIN).sum()), len(_am),
                 int(((_am >= _G.ANA_MIN) & (_am - _lx >= _G.MARGIN_MIN)).sum())), flush=True)

    cand = [(m, v) for m, v in PC.items() if v.get("pc") and str(m) in Z]
    if C3:
        # every candidate molecule is offered to the Class-3 gate, which declines cheaply in the
        # worker.  Firing on nothing is a legal outcome -- that is the point of replacing the rank
        # gate, which always fired on exactly MUT_FRAC of the field.
        fire = [m for m, _ in cand]
        print(f"gate: c3_gate over all {len(fire)} candidate molecules "
              f"(analog + exact-match + depth, thresholds in c3_gate)", flush=True)
    else:
        scored = [(float(v["best_pool_fz"]) if np.isfinite(v.get("best_pool_fz", np.nan)) else -np.inf, m)
                  for m, v in cand]
        scored.sort()
        n_fire = max(1, int(round(MUT_FRAC * len(scored))))
        fire = [m for _, m in scored[:n_fire]]
        print(f"gate: firing on {len(fire)} of {len(cand)} molecules "
              f"(MUT_FRAC={MUT_FRAC}, weakest best_pool_fz)", flush=True)
    if MAXM:
        fire = fire[:MAXM]

    tasks = [(m, Z[str(m)], PC[m]["pc"], PC[m].get("pc_keys", []),
              (EN.get(str(m)) or {}).get("smiles", []), (EN.get(str(m)) or {}).get("keys", []),
              C3.get(str(m)) if C3 else None)
             for m in fire]
    res, diags = {}, []
    import multiprocessing as mp
    ctx = mp.get_context("fork" if sys.platform != "win32" else "spawn")
    with ctx.Pool(NW, initializer=init_worker, initargs=(V3CODE, POOL_DIR, MUT_DIR)) as P:
        for k, out in enumerate(P.imap_unordered(mutate_one, tasks, chunksize=1)):
            mid, smis, fzs, keys, tail = out
            diag, forms, prov = tail if isinstance(tail, tuple) else (tail, [], [])
            diags.append(diag)
            if smis:
                res[str(mid)] = dict(smiles=smis, keys=keys, formula=forms, fz=fzs, src=prov,
                                     tier=diag.get("tier") or "")
            if (k + 1) % 20 == 0 or k + 1 == len(tasks):
                print(f"  mutator {k+1}/{len(tasks)} {time.time()-T0:.0f}s", flush=True)

    json.dump(res, open(OUT, "w"))
    json.dump(diags, open(OUT + ".diag.json", "w"))
    tot = sum(len(v["smiles"]) for v in res.values())
    gen = sum(d.get("n_gen", 0) for d in diags)
    print(f"mut_runner done: {len(res)} molecules with candidates, {tot} candidates listed, "
          f"{gen} structures enumerated, {time.time()-T0:.0f}s", flush=True)
    print(f"MUT SOURCES: scaffolds pc={sum(d.get('n_sc_pc', 0) for d in diags)} "
          f"eng={sum(d.get('n_sc_eng', 0) for d in diags)} "
          f"ana={sum(d.get('n_sc_ana', 0) for d in diags)} | "
          f"eng-only={sum(d.get('n_sc_eng_only', 0) for d in diags)} "
          f"ana-only={sum(d.get('n_sc_ana_only', 0) for d in diags)} | "
          f"candidates: eng={sum(d.get('n_listed_eng', 0) for d in diags)} "
          f"ana={sum(d.get('n_listed_ana', 0) for d in diags)} | "
          f"pairs refused as too deep={sum(d.get('pairs_too_deep', 0) for d in diags)} | "
          f"deadline hit on {sum(d.get('hit_deadline', 0) for d in diags)} molecules", flush=True)
    if C3:
        import collections as _c
        tiers = _c.Counter(d.get("tier") or "declined" for d in diags)
        why = _c.Counter(d.get("gate") or "" for d in diags if not d.get("tier"))
        print(f"C3 GATE: fired on {len(diags) - tiers['declined']} of {len(diags)} | "
              f"sure={tiers['sure']} likely={tiers['likely']} declined={tiers['declined']}",
              flush=True)
        print("C3 GATE REASONS: " + " ".join(f"{k}={v}" for k, v in why.most_common() if k),
              flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
