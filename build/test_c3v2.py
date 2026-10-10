"""Tests for the c3v2 additions: delta_m_mutator (v2 section) and c3_gate (stage1_v2,
classify_v2, cap_fires).  Run: python build/test_c3v2.py

The v1 behaviour is covered by test_delta_m_mutator.py / test_c3_gate.py, which must keep passing
unchanged -- the first test here checks the v1 defaults were not disturbed.
"""
import collections
import importlib
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from rdkit import Chem
from rdkit.Chem import inchi

import c3_gate as G
import delta_m_mutator as M

FAIL = []

Q = "O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12"                       # quercetin
QG = "OCC1OC(Oc2c(-c3ccc(O)c(O)c3)oc3cc(O)cc(O)c3c2=O)C(O)C(O)C1O"    # quercetin 3-O-glucoside
ISO = "COc1cc(-c2oc3cc(O)cc(O)c3c(=O)c2O)ccc1O"                       # isorhamnetin (3'-OMe)


def check(cond, msg):
    if cond:
        print(f"  ok   {msg}")
    else:
        print(f"  FAIL {msg}")
        FAIL.append(msg)


def ik(s):
    return inchi.MolToInchiKey(Chem.MolFromSmiles(s))[:14]


R2 = M.build_reach_signed(2)
IDX = M.mass_index(R2)


# ------------------------------------------------------------------ mutator
def test_v1_defaults_untouched():
    print("test_v1_defaults_untouched")
    check(len(M.PALETTE) == 20, "v1 palette still has 20 transforms")
    check(len(M.build_reach()) == 114854, "v1 reach table unchanged (114,854 deltas)")
    check(not set(M.EXTRA_PALETTE) & set(M.PALETTE), "v2 extras live outside PALETTE")


def test_every_v2_move_matches_its_vector():
    print("test_every_v2_move_matches_its_vector")
    probes = ["CCC(C)Cc1ccccc1O", "CCCC(O)C", "C=CC(=O)NCC=O", "CNc1ccccc1C=C", QG, ISO,
              "CC(=O)Oc1ccccc1", "CC(O)CC=CC",
              # one carrier per reverse move, so every inverse is exercised, not just the common ones
              "OS(=O)(=O)Oc1ccccc1", "OP(=O)(O)Oc1ccccc1", "Brc1ccc(I)cc1F", "Clc1ccccc1S",
              "NC(=O)c1ccccc1CNC", "OC(=O)C1OC(Oc2ccccc2)C(O)C(O)C1O", "OC1COC(Oc2ccccc2)C(O)C1O",
              "CC1OC(Oc2ccccc2)C(O)C(O)C1O", "OC(=O)CC(=O)Oc1ccccc1", "O=C(C=Cc1ccc(O)cc1)Oc1ccccc1",
              "O=C(C=Cc1ccc(O)c(O)c1)Oc1ccccc1", "COc1cc(C=CC(=O)Oc2ccccc2)ccc1O",
              "O=C(Oc1ccccc1)c1cc(O)c(O)c(O)c1", "CC(C)=CCc1ccccc1", "CC(C)=CCOc1ccccc1",
              "Nc1ccccc1", "O=Cc1ccccc1", "CC(=O)c1ccccc1", "OCc1ccccc1"]
    bad, used = [], set()
    names = M.signed_names()
    for n in names:
        for s in probes:
            m = Chem.MolFromSmiles(s)
            v0 = M.elem_vec(m)
            for p in M._run_move(m, n):
                try:
                    Chem.SanitizeMol(p)
                except Exception:
                    continue
                used.add(n)
                d = tuple(a - b for a, b in zip(M.elem_vec(p), v0))
                if d != M.signed_vec(n):
                    bad.append((n, s))
    check(not bad, "every product of every signed move differs from its input by the move's vector")
    check(used == set(names), "and every one of the %d moves fired on some probe (missing: %s)"
          % (len(names), sorted(set(names) - used)))


def test_excise_inverses():
    print("test_excise_inverses")
    for src, move, want, lab in [(QG, "-hexose", Q, "glucoside -> aglycone"),
                                 (ISO, "-CH2", Q, "methyl ether -> phenol"),
                                 ("CC(=O)Oc1ccccc1", "-C2H2O", "Oc1ccccc1", "acetate -> phenol"),
                                 ("Clc1ccccc1", "-Cl", "c1ccccc1", "aryl chloride -> arene")]:
        prods = [Chem.MolToSmiles(p) for p in M._run_move(Chem.MolFromSmiles(src), move)]
        check(ik(want) in {ik(p) for p in prods}, "%s via %s" % (lab, move))
        check(all("." not in p for p in prods), "  and no disconnected product")


def test_signed_reach():
    print("test_signed_reach")
    inv_pair = 0
    for decs in R2.values():
        for d in decs:
            if any(M.inverse_name(n) in d for n in d):
                inv_pair += 1
    check(inv_pair == 0, "no decomposition holds a move together with its own inverse")
    check(tuple([0] * len(M.EL)) in R2, "identity present at depth 0")
    hex_v = M.ADDITIVE_V2["+hexose"][0]
    check(("-hexose",) in R2.get(M._neg(hex_v), ()), "the negative hexose delta is a 1-step reverse move")


def test_match_mass_delta():
    print("test_match_mass_delta")
    hex_v = M.ADDITIVE_V2["+hexose"][0]
    got = M.match_mass_delta(M.vec_mass(hex_v), IDX, 0.003)
    check(got and got[0] == hex_v, "162.0528 Da resolves to C6H10O5 first")
    check(M._neg(hex_v) in M.match_mass_delta(-M.vec_mass(hex_v), IDX, 0.003),
          "and -162.0528 to the reverse delta")
    check(not M.match_mass_delta(0.5, IDX, 0.003), "an arbitrary mass matches nothing")
    check(len(M.match_mass_delta(14.0157, IDX, 0.003)) >= 1, "CH2 (14.0157) matches")


def _roundtrip(a, b, **kw):
    dm = M.vec_mass(M.vec_of_smiles(b)) - M.vec_mass(M.vec_of_smiles(a))
    out = M.mutate_anchored([(a, M.match_mass_delta(dm, IDX, 0.003))], R2, beam=200, cap=2000, **kw)
    return [s for s, _, _ in out]


def test_anchored_roundtrips():
    print("test_anchored_roundtrips")
    for a, b, lab in [(QG, Q, "deglycosylation"), (ISO, Q, "demethylation"),
                      (Q, QG, "glycosylation"), (Q, ISO, "O-methylation")]:
        out = _roundtrip(a, b)
        check(ik(b) in {ik(s) for s in out}, "%s: truth regenerated from the exact mass delta" % lab)
        tv = M.vec_of_smiles(b)
        check(all(M.vec_of_smiles(s) == tv for s in out), "  every product sits at the target formula")


def _oracle(truth):
    from rdkit.Chem import rdFingerprintGenerator
    g = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    z = 2.0 * g.GetFingerprintAsNumPy(Chem.MolFromSmiles(truth)).astype(float) - 1.0
    return lambda smis: np.array([float(g.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) @ z)
                                  for s in smis])


def test_pruning():
    print("test_pruning")
    sc = _oracle(QG)
    full = _roundtrip(Q, QG)
    loose = _roundtrip(Q, QG, scorer=sc, tau=1e9)
    check(set(map(ik, loose)) == set(map(ik, full)), "tau=+inf prunes nothing")
    tight = _roundtrip(Q, QG, scorer=sc, tau=0.0)
    check(ik(QG) in set(map(ik, tight)), "tau=0 keeps the truth (it beats the anchor under oracle z)")
    check(len(tight) <= len(full), "and never adds candidates")
    st = collections.Counter()
    none = M.mutate_anchored([(Q, [M.ADDITIVE_V2["+hexose"][0]])], R2, scorer=sc, tau=-1e9, stats=st)
    check(not none and st["pruned"] > 0, "a floor above every score prunes everything, and counts it")


def test_all_orders_tried():
    print("test_all_orders_tried")
    # +CH2 can only O-methylate once +O has made the hydroxyl: only one of the two orders gives
    # anisole, so a single fixed order of a dependent pair would miss it.
    out = M.grow_signed(Chem.MolFromSmiles("c1ccccc1"), ("+CH2", "+O"),
                        M.vec_of_smiles("COc1ccccc1"), beam=200)
    check(ik("COc1ccccc1") in {ik(s) for s in out}, "anisole reached from benzene via +O then +CH2")


# ------------------------------------------------------------------ gate
S1 = dict(lib_exact=0.30, exact_max=0.20, ana_max=0.85, delta_steps=1, fz_best=12.0,
          fz_anchor=8.0, best_pool_fz=10.0, fz_top=11.0, fz_sep=2.0)


def s1(**kw):
    d = dict(S1)
    d.update(kw)
    return G.stage1_v2(**d)


def test_stage1():
    print("test_stage1")
    check(s1() == G.SURE, "everything agrees -> sure")
    check(s1(fz_best=8.0) is None, "no f.z improvement over the anchor -> declined (I0)")
    check(s1(best_pool_fz=12.5) is None, "a pool structure fits the fingerprint better -> declined")
    check(s1(fz_top=13.0) == G.LIKELY, "PubChem-only top beats it -> at most likely")
    check(s1(fz_sep=None) == G.LIKELY, "fewer than 3 candidates (no separation) -> at most likely")
    check(s1(fz_sep=0.5) == G.LIKELY, "best edit does not stand out from its regioisomers -> likely")
    check(s1(lib_exact=0.95) is None, "library match -> declined")
    check(s1(delta_steps=3) is None, "delta deeper than MAX_STEPS_V2 -> declined")
    check(s1(delta_steps=None) is None, "no anchor mass delta matched -> declined")
    check(s1(ana_max=0.4) is None, "no analog evidence -> declined")
    r = collections.Counter()
    s1(fz_best=8.0, reasons=r)
    check(r["no_fp_improvement"] == 1, "reasons name the veto")
    check(s1(fz_best=1e6, fz_anchor=1e6 - 1, best_pool_fz=1e6 - 2, fz_top=1e6 - 3) == G.SURE,
          "only within-molecule differences matter: shifting every f.z by 1e6 changes nothing")


def test_stage2():
    print("test_stage2")
    c = lambda **kw: G.classify_v2(**{**dict(tier1=G.SURE, lib_max=0.3, top1_agree=False,
                                             gen_ice=0.5, ret_ice=0.4, gen_gl=0.5, ret_gl=0.4), **kw})
    check(c() == G.SURE, "FPNet, ICE, GL agree and the engines disagree -> sure")
    check(c(lib_max=0.95) is None, "real v4n lib_max vetoes")
    check(c(ret_ice=G.ICE_MAX) is None, "a retrieved candidate the simulator reproduces vetoes")
    check(c(ret_ice=G.ICE_MAX, gen_ice=G.ICE_MAX + 0.1, ice_unit=False) is not None,
          "but not when the ICE scale is unverified (absolute cut skipped)")
    check(c(gen_ice=0.30, ret_ice=0.40) is None, "ICE clearly prefers the retrieved top-1 -> declined")
    check(c(gen_ice=0.38, ret_ice=0.40) == G.LIKELY, "within ICE_SLACK -> kept, but not sure")
    check(c(top1_agree=True) == G.LIKELY, "both engines agree on top-1 -> never rank 1")
    check(c(gen_gl=0.3) == G.LIKELY, "GLACIER disagrees -> demoted")
    check(c(gen_gl=None) == G.SURE, "GLACIER absent ([M+H]+ only) does not block")
    check(c(gen_ice=None) == G.LIKELY, "ICE did not score it (budget) -> demoted, not promoted")
    check(c(tier1=None) is None, "stage-1 decline is final")
    check(c(tier1=G.LIKELY) == G.LIKELY, "stage 2 never upgrades")
    os.environ["C3_ALLOW_RANK1"] = "0"
    try:
        H = importlib.reload(G)
        check(H.classify_v2(H.SURE, 0.3, False, 0.5, 0.4, 0.5, 0.4) == H.LIKELY,
              "C3_ALLOW_RANK1=0 turns every sure into likely")
    finally:
        del os.environ["C3_ALLOW_RANK1"]
        importlib.reload(G)


def test_cap():
    print("test_cap")
    rows = [("m%d" % i, G.SURE if i < 20 else (G.LIKELY if i < 40 else None), 1.0 - i / 100)
            for i in range(400)]
    out, st = G.cap_fires(rows, 400)
    n_sure = sum(1 for v in out.values() if v == G.SURE)
    n_fire = sum(1 for v in out.values() if v)
    check(n_sure == int(G.SURE_FRAC * 400), "sure capped at SURE_FRAC (%d)" % n_sure)
    check(n_fire == min(40, int(G.FIRE_FRAC * 400)), "fires capped at FIRE_FRAC (%d)" % n_fire)
    check(out["m0"] == G.SURE and out["m19"] == G.LIKELY, "the most confident keep sure")
    storm = [("m%d" % i, G.LIKELY, 0.5) for i in range(150)] + [("x%d" % i, None, 0) for i in range(250)]
    out2, st2 = G.cap_fires(storm, 400)
    check(st2["disabled"] == 1 and not any(out2.values()),
          "firing on >25% of the run switches the channel off (bio2 protection)")
    out3, st3 = G.cap_fires([("a", None, 0)] * 10, 400)
    check(not any(out3.values()) and not st3["disabled"], "firing on nothing is a legal outcome")


if __name__ == "__main__":
    for fn in [test_v1_defaults_untouched, test_every_v2_move_matches_its_vector,
               test_excise_inverses, test_signed_reach, test_match_mass_delta,
               test_anchored_roundtrips, test_pruning, test_all_orders_tried,
               test_stage1, test_stage2, test_cap]:
        fn()
    print()
    if FAIL:
        print(f"{len(FAIL)} FAILED")
        for f in FAIL:
            print("  -", f)
        sys.exit(1)
    print("all tests passed")
