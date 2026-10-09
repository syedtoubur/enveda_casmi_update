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
# ---- v2 (Track 4 c3v2): anchored whole-molecule mutation, signed moves, exact-mass delta
MUT_V2 = os.environ.get("MUT_V2", "") == "1"
MUT_ANCHORS = int(os.environ.get("MUT_ANCHORS", 5))     # analogs whose mass delta matches, best first
MUT_DM_TOL = float(os.environ.get("MUT_DM_TOL", 0.003))  # Da, anchor -> query exact mass delta
MUT_TAU = os.environ.get("MUT_TAU", "")                 # I0b prune floor in mean-|z| units; "" = off
#   tau is scale-free: the floor is score(anchor) - MUT_TAU * mean(|z|), so it means "an intermediate
#   may lose about MUT_TAU bits' worth of agreement".  The local oracle (eval_anchored_oracle.py, one
#   bit ~ 1) lost 0 truths at tau=8 and 2.3% at tau=2-4 on depth-2 round trips.
MUT_BETA = float(os.environ.get("MUT_BETA", 0.0))       # I0c consensus-core weight; 0 = off
MUT_BEAM = int(os.environ.get("MUT_BEAM", 200))
PPM_CENTER = float(os.environ.get("MUT_PPM_CENTER", 1.45))   # timsTOF precursor offset (CLAUDE.md)
MUT_KEEP_DECLINED = os.environ.get("MUT_KEEP_DECLINED", "") == "1"   # calibration only: emit lists of
#   molecules stage 1 declined (tier "") so generator recall can be measured apart from the gate

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
    if MUT_V2:
        _W["reach2"] = dm.build_reach_signed(max(2, _W["c3"].MAX_STEPS_V2))
        _W["midx"] = dm.mass_index(_W["reach2"])
    ban = os.environ.get("CASMI_BAN_KEYS", "")       # calibration only: cohort truths removed from the
    if ban and os.path.exists(ban):                  # pool, so a regenerated truth is not dropped as known
        _W["pool_keys"] -= set(json.load(open(ban)).get("pool_ban", []))


def _fz_fn(z):
    """f(list of SMILES) -> f.z with the full selected-bit fingerprint, cached.  This is the same
    `fp[bits] @ z` the PubChem channel uses for fz_top, and best_pool_fz uses the pool's packed copy
    of the same fingerprint, so all three are on one scale for a given molecule."""
    chem, bits = _W["chem"], _W["bits"]
    z = np.asarray(z, np.float32)
    cache = {}

    def f(smiles):
        out = np.full(len(smiles), -np.inf)
        for i, s in enumerate(smiles):
            if s not in cache:
                fp = chem.raw_fingerprint(s)
                cache[s] = -np.inf if fp is None else float(fp[bits].astype(np.float32) @ z)
            out[i] = cache[s]
        return out
    f.cache = cache
    return f


def consensus_core(neigh, power=3.0, min_n=3, min_sim=0.5, p_core=0.8):
    """I0c: conserved-present bits over the analog neighbourhood -> (bit index array, logit weights).

    One-sided on purpose.  The anchors are inside this consensus, so a symmetric prior would score
    every edit below its own anchor; only *keeping* bits the family conserves is rewarded, and bits
    no neighbour carries score 0 so a novel decoration costs nothing."""
    chem, bits = _W["chem"], _W["bits"]
    rows, w = [], []
    for smi, sim in neigh:
        if sim < min_sim:
            continue
        fp = chem.raw_fingerprint(smi)
        if fp is not None:
            rows.append(fp[bits].astype(np.float64))
            w.append(float(sim) ** power)
    if len(rows) < min_n:
        return np.zeros(0, int), np.zeros(0)
    w = np.asarray(w) / np.sum(w)
    p = (np.stack(rows) * w[:, None]).sum(0)
    idx = np.where(p >= p_core)[0]
    pc = np.clip(p[idx], 1e-3, 0.99)
    return idx, np.log(pc / (1 - pc))


def mutate_one_v2(task):
    """task = (mid, z, pc_row, pc_keys, eng_keys, c3) -> (mid, smiles, f.z, keys, (diag, forms, src)).

    Anchors are engine-A shifted analogs whose exact mass delta to the query matches a signed move
    combination (I1/I3).  Cheap vetoes run before generation; the f.z tests (I0) after it."""
    import collections
    mid, z, pc_row, pc_keys, eng_keys, c3 = task
    t0 = time.time()
    chem, dm, G = _W["chem"], _W["dm"], _W["c3"]
    diag = dict(molecule_id=mid, n_gen=0, n_listed=0, tier=None, v2=1)
    reasons, st = collections.Counter(), collections.Counter()

    def done(tier=None, out=(), fzs=(), keys=(), forms=(), srcs=()):
        diag.update(tier=tier, gate=(list(reasons)[-1] if reasons and not tier else ""),
                    secs=round(time.time() - t0, 2), **{k: int(v) for k, v in st.items()})
        return mid, list(out), list(fzs), list(keys), (diag, list(forms), list(srcs))

    ana = (c3 or {}).get("ana") or []
    exact = float((c3 or {}).get("lib_exact") or 0.0)
    target = float(pc_row.get("target") or (c3 or {}).get("target") or 0.0)
    target *= 1.0 - PPM_CENTER * 1e-6
    anchors = []
    for smi, mass, sim in ana:
        dms = dm.match_mass_delta(target - float(mass), _W["midx"], MUT_DM_TOL)
        dms = [d for d in dms if min(len(x) for x in _W["reach2"][d]) <= G.MAX_STEPS_V2]
        if dms:
            anchors.append((smi, float(sim), dms))
    anchors.sort(key=lambda a: -a[1])
    anchors = anchors[:MUT_ANCHORS]
    ana_max = anchors[0][1] if anchors else (float(ana[0][2]) if ana else 0.0)
    steps = (min(len(x) for d in anchors[0][2] for x in _W["reach2"][d]) if anchors else None)
    diag.update(ana_max=round(ana_max, 4), lib_exact=round(exact, 4), n_anchors=len(anchors),
                delta_steps=(-1 if steps is None else steps))
    # cheap vetoes first: fz_best/fz_anchor are not known yet, so pass sentinels that cannot veto
    if G.stage1_v2(exact, exact, ana_max, steps, 1.0, 0.0, None, reasons=reasons) is None:
        return done()
    reasons.clear()

    score = _fz_fn(z)
    core_idx, core_w = (consensus_core([(a[0], a[2]) for a in ana]) if MUT_BETA > 0
                        else (np.zeros(0, int), np.zeros(0)))
    if MUT_BETA > 0 and len(core_idx):
        bits = _W["bits"]

        def scorer(smiles):
            base = score(smiles)
            bonus = np.zeros(len(smiles))
            for i, s in enumerate(smiles):
                fp = chem.raw_fingerprint(s)
                if fp is not None:
                    bonus[i] = float(fp[bits][core_idx].astype(np.float64) @ core_w)
            return base + MUT_BETA * bonus
    else:
        scorer = score
    diag["core_bits"] = int(len(core_idx))
    tau = (float(MUT_TAU) * float(np.mean(np.abs(np.asarray(z, np.float64))))
           if MUT_TAU != "" else None)
    try:
        gen = dm.mutate_anchored([(a[0], a[2]) for a in anchors], _W["reach2"], beam=MUT_BEAM,
                                 cap=MUT_CAP, max_steps=G.MAX_STEPS_V2,
                                 scorer=scorer if tau is not None else None, tau=tau,
                                 deadline=time.monotonic() + MUT_DEADLINE, stats=st)
    except Exception as e:
        diag["error"] = repr(e)[:200]
        return done()
    diag["n_gen"] = len(gen)
    if not gen:
        reasons["no_candidates"] += 1
        return done()

    smis = [s for s, _, _ in gen]
    fz = score(smis)
    rank_sc = scorer(smis) if scorer is not score else fz
    fz_anchor = {i: float(score([a[0]])[0]) for i, a in enumerate(anchors)}
    known = set(pc_keys) | set(eng_keys) | _W["pool_keys"]
    out, out_fz, out_k, out_f, out_s, seen = [], [], [], [], [], set()
    best_anchor = None
    for o in np.argsort(-rank_sc, kind="stable"):
        s, ai, dec = gen[o]
        k = chem.score_key(s)
        if k is None or k in seen:
            continue
        seen.add(k)
        if k in known:
            st["in_db"] += 1                    # a database structure: retrieval already has it
            continue
        if best_anchor is None:
            best_anchor = ai
        out.append(s)
        out_fz.append(float(fz[o]))
        out_k.append(k)
        out_f.append(dm.fmt_vec(dm.vec_of_smiles(s)) + "|gen")
        out_s.append("a%d:%s" % (ai, "+".join(dec)))
        if len(out) >= MUT_K:
            break
    if not out:
        reasons["no_new_candidates"] += 1
        return done()
    fin = fz[np.isfinite(fz)]
    sep = (float((out_fz[0] - np.median(fin)) / (np.std(fin) + 1e-9)) if len(fin) >= 3 else None)
    bp = pc_row.get("best_pool_fz")
    bp = float(bp) if bp is not None and np.isfinite(float(bp)) else None
    ft = pc_row.get("fz_top")
    ft = float(ft) if ft is not None else None
    tier = G.stage1_v2(exact, exact, anchors[best_anchor][1], steps, out_fz[0],
                       fz_anchor[best_anchor], bp, ft, sep, reasons=reasons)
    diag.update(fz_best=round(out_fz[0], 3), fz_anchor=round(fz_anchor[best_anchor], 3),
                best_pool_fz=(None if bp is None else round(bp, 3)),
                fz_top=(None if ft is None else round(ft, 3)),
                fz_sep=(None if sep is None else round(sep, 3)), n_listed=len(out))
    if tier is None:
        if MUT_KEEP_DECLINED:
            return done(None, out, out_fz, out_k, out_f, out_s)
        return done()
    return done(tier, out, out_fz, out_k, out_f, out_s)


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

    # v2 anchors on spectral analogs, not on the PubChem list, so an empty PubChem list (a plausible
    # Class-3 situation) does not exclude the molecule; the v1 paths keep their exact behaviour.
    cand = [(m, v) for m, v in PC.items() if (v.get("pc") or MUT_V2) and str(m) in Z]
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

    if MUT_V2:
        # v2 needs the engine diagnostics (anchors) for every molecule; no C3 -> nothing to anchor
        fire = [m for m in fire if str(m) in C3]
        _row = lambda m: {k: PC[m].get(k) for k in ("best_pool_fz", "fz_top", "target")}
        tasks = [(m, Z[str(m)], _row(m), PC[m].get("pc_keys", []),
                  (EN.get(str(m)) or {}).get("keys", []), C3.get(str(m))) for m in fire]
        worker = mutate_one_v2
        print(f"MUT V2: anchored signed mutation on {len(tasks)} molecules | anchors<={MUT_ANCHORS} "
              f"dm_tol={MUT_DM_TOL} tau={MUT_TAU or 'off'} beta={MUT_BETA} beam={MUT_BEAM}",
              flush=True)
    else:
        tasks = [(m, Z[str(m)], PC[m]["pc"], PC[m].get("pc_keys", []),
                  (EN.get(str(m)) or {}).get("smiles", []), (EN.get(str(m)) or {}).get("keys", []),
                  C3.get(str(m)) if C3 else None)
                 for m in fire]
        worker = mutate_one
    res, diags = {}, []
    import multiprocessing as mp
    ctx = mp.get_context("fork" if sys.platform != "win32" else "spawn")
    with ctx.Pool(NW, initializer=init_worker, initargs=(V3CODE, POOL_DIR, MUT_DIR)) as P:
        for k, out in enumerate(P.imap_unordered(worker, tasks, chunksize=1)):
            mid, smis, fzs, keys, tail = out
            diag, forms, prov = tail if isinstance(tail, tuple) else (tail, [], [])
            diags.append(diag)
            if smis:
                res[str(mid)] = dict(smiles=smis, keys=keys, formula=forms, fz=fzs, src=prov,
                                     tier=diag.get("tier") or "")
                if MUT_V2:      # stage 2 and the cap read these in the fusion cell
                    res[str(mid)]["sig"] = {k: diag.get(k) for k in (
                        "ana_max", "lib_exact", "delta_steps", "fz_best", "fz_anchor",
                        "best_pool_fz", "fz_top", "fz_sep")}
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
    if MUT_V2:
        _sum = lambda k: sum(int(d.get(k, 0) or 0) for d in diags)
        _imp = [d["fz_best"] - d["fz_anchor"] for d in diags
                if d.get("fz_best") is not None and d.get("fz_anchor") is not None]
        print(f"MUT V2 STATS: anchors>0 on {sum(1 for d in diags if d.get('n_anchors'))} | "
              f"generated {_sum('n_gen')} | reverse moves used {_sum('used_reverse')} | "
              f"pruned {_sum('pruned')} | in-db dropped {_sum('in_db')} | "
              f"deadline {_sum('hit_deadline')} | core bits med "
              f"{int(np.median([d.get('core_bits', 0) for d in diags])) if diags else 0}", flush=True)
        if _imp:
            print("MUT V2 fz_best - fz_anchor (percentiles 10/50/90): "
                  + " ".join("%.2f" % x for x in np.percentile(_imp, [10, 50, 90])), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
