"""Tests for build/delta_m_mutator.py (specs_v4n_update M2). Run: python build/test_delta_m_mutator.py"""
import collections
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

import delta_m_mutator as M

RDLogger.DisableLog("rdApp.*")
FAIL = []


def check(cond, msg):
    if cond:
        print(f"  ok   {msg}")
    else:
        print(f"  FAIL {msg}")
        FAIL.append(msg)


# a substrate carrying every socket the palette needs: aromatic CH, aliphatic CH2, OH, NH, C=C
PROBE = "Oc1ccccc1CCNC=C"


def test_palette_vectors_match_smarts():
    """Every declared formula delta must equal what its own SMARTS actually does to a molecule.

    This is the test that caught +PO3: R-OH -> R-O-PO(OH)2 adds P, 3 O *and* one H, so the vector
    is P1 O3 H1, not P1 O3.
    """
    print("test_palette_vectors_match_smarts")
    base = Chem.MolFromSmiles(PROBE)
    bv = M.elem_vec(base)
    for name, (vec, smarts) in M.PALETTE.items():
        fired = False
        for sm in smarts:
            rxn = AllChem.ReactionFromSmarts(sm)
            for prod in rxn.RunReactants((base,)):
                q = prod[0]
                try:
                    Chem.SanitizeMol(q)
                except Exception:
                    continue
                pv = M.elem_vec(q)
                if pv is None:
                    continue
                got = tuple(a - b for a, b in zip(pv, bv))
                check(got == vec, f"{name}: {sm} gives {M.fmt_vec(got) or '0'}, declared {M.fmt_vec(vec) or '0'}")
                fired = True
                break
            if fired:
                break
        if not fired:
            check(False, f"{name}: no SMARTS fired on the probe molecule")


def test_reach_table():
    print("test_reach_table")
    R = M.build_reach(maxops=2, max_dec=4)
    check(R[tuple([0] * len(M.EL))] == ((),), "empty delta maps to the empty decomposition")
    check(R[M.PALETTE["+O"][0]][0] == ("+O",), "single +O resolves to itself")
    d = tuple(a + b for a, b in zip(M.PALETTE["+O"][0], M.PALETTE["+CH2"][0]))
    check(d in R, "+O plus +CH2 is reachable at depth 2")
    check(all(len(v) <= 4 for v in R.values()), "no delta keeps more than max_dec decompositions")
    check(all(len(v[0]) <= len(v[-1]) for v in R.values()), "decompositions are stored shortest-first")
    # the methoxy ambiguity that motivates max_dec > 1
    dm = M.PALETTE["+CH2O"][0]
    check(len(R.get(dm, ())) >= 2, "+CH2O delta keeps both the direct and the +O/+CH2 route")


def test_elem_vec_and_formula():
    print("test_elem_vec_and_formula")
    v = M.vec_of_smiles("c1ccccc1")
    check(v == M._v(C=6, H=6), f"benzene is C6H6, got {M.fmt_vec(v)}")
    check(M.vec_of_smiles("[Si]") is None, "an out-of-palette element returns None")
    check(M.vec_of_smiles("not a smiles") is None, "an unparseable SMILES returns None")


def test_harvest():
    print("test_harvest")
    hits = ["Oc1ccc2c(c1)oc(-c1ccc(O)cc1)c(O)c2=O", "CCCCCCCC(=O)O", "c1ccccc1"]
    sc = M.harvest_scaffolds(hits)
    check(len(sc) >= 2, f"distinct scaffolds harvested, got {len(sc)}")
    oct_ = Chem.MolToSmiles(Chem.MolFromSmiles("CCCCCCCC(=O)O"))
    check(oct_ in sc, "an acyclic hit falls back to the whole structure")
    fs = M.harvest_formulas(hits)
    check(len(fs) == 3 and all(isinstance(f, tuple) for f in fs), f"formula vectors returned, got {len(fs)}")


def test_grow_formula_exact():
    print("test_grow_formula_exact")
    R = M.build_reach(maxops=3)
    scaf = Chem.MolFromSmiles("c1ccc2ccccc2c1")
    sv = M.elem_vec(scaf)
    target = tuple(a + b for a, b in zip(sv, M.PALETTE["+O"][0]))
    got = M.grow(scaf, R[M.PALETTE["+O"][0]][0], target)
    check(len(got) > 0, f"hydroxylating naphthalene yields products, got {len(got)}")
    check(all(M.vec_of_smiles(s) == target for s in got), "every product matches the target formula exactly")
    check(len(set(got)) == len(got), "products are deduplicated")


def test_socket_order():
    print("test_socket_order")
    check(M._order(("+hexose", "+O")) == ["+O", "+hexose"], "+O is applied before an [OX2H]-gated transform")
    # a Murcko scaffold has no O-H, so the sugar can only attach to an oxygen +O just made
    scaf = Chem.MolFromSmiles("c1ccc2ccccc2c1")
    d = tuple(a + b for a, b in zip(M.PALETTE["+O"][0], M.PALETTE["+hexose"][0]))
    target = tuple(a + b for a, b in zip(M.elem_vec(scaf), d))
    got = M.grow(scaf, ("+O", "+hexose"), target)
    check(len(got) > 0, f"glycosylation reachable after hydroxylation, got {len(got)}")
    check(len(M.grow(scaf, ("+hexose", "+O"), target)) > 0, "the reverse input order is reordered, not lost")


def test_mutate_end_to_end():
    print("test_mutate_end_to_end")
    R = M.build_reach()
    hits = ["COc1cc(C2Oc3cc(O)ccc3C2)ccc1O", "Oc1ccc2c(c1)oc(-c1ccc(O)cc1)c(O)c2=O"]
    out = M.mutate(hits, R)
    check(len(out) > 0, f"candidates produced, got {len(out)}")
    check(len(set(out)) == len(out), "output is deduplicated")
    check(all(Chem.MolFromSmiles(s) is not None for s in out[:50]), "every product parses")
    allowed = set(M.harvest_formulas(hits))
    check(all(M.vec_of_smiles(s) in allowed for s in out[:50]), "every product sits at one of the hits' formulas")
    capped = M.mutate(hits, R, cap=10)
    check(len(capped) <= 10, f"the product cap is respected, got {len(capped)}")
    check(M.mutate([], R) == [], "no hits means no candidates")
    check(M.mutate(["garbage"], R) == [], "unparseable hits are skipped")




# ------------------------------------------------------------------ multi-source harvest (bio)
#   The mutator-bio notebook adds engine-A's top-40 as a second scaffold source. These tests pin
#   the two properties that make that a single-variable change: a single source behaves exactly as
#   before, and neither source can starve the other.
A_HITS = ["Oc1ccc2c(c1)oc(-c1ccc(O)cc1)c(O)c2=O", "CC(=O)Oc1ccccc1C(=O)O"]
B_HITS = ["c1ccc2ccccc2c1", "c1ccc2[nH]ccc2c1"]
# naphthalene + 2-naphthol + 2,6-dihydroxynaphthalene: one Murcko scaffold (naphthalene) and
# three target formulas -- C10H8 (identity), C10H8O (+O, one step), C10H8O2 (+2O, two steps).
# Shallow on purpose: A_HITS needs six transforms to bridge scaffold to target, so a depth cap
# empties it and every comparison below would be vacuous.
SHALLOW = ["c1ccc2ccccc2c1", "Oc1ccc2ccccc2c1", "Oc1ccc2cc(O)ccc2c1"]
AZ_HITS = ["C1=CC2=CC=CC=CC2=C1"]      # Murcko scaffold azulene: another C10H8 ring system
NAPH = Chem.MolToSmiles(Chem.MolFromSmiles("c1ccc2ccccc2c1"))


def test_single_source_equivalence():
    """mutate() must stay byte-identical, so the bio notebook differs only by the scaffold source."""
    print("test_single_source_equivalence")
    R = M.build_reach(maxops=3)
    hits = ["COc1cc(C2Oc3cc(O)ccc3C2)ccc1O", "Oc1ccc2c(c1)oc(-c1ccc(O)cc1)c(O)c2=O"]
    a = M.mutate(hits, R)
    b, prov = M.mutate_sources([("pc", hits)], R, per_source=(25,))
    check(a == b, f"the wrapper and a one-source call agree exactly ({len(a)} vs {len(b)})")
    check(set(prov) == set(b), "every candidate carries provenance")
    check(set(prov.values()) <= {"pc"}, "one source means one provenance label")
    c, _ = M.mutate_sources([("pc", hits)], R)
    check(c == a, "the default per_source=(25, 40) leaves a single source unchanged")
    d, _ = M.mutate_sources([("pc", hits), ("eng", [])], R)
    check(d == a, "an empty second source is dropped, not walked")


def test_per_source_limit_trap():
    """harvest_scaffolds truncates at `limit`; concatenating the sources first throws one away."""
    print("test_per_source_limit_trap")
    flat = M.harvest_scaffolds(B_HITS[:1] * 25 + A_HITS[:1], 25)
    check(len(flat) == 1, "concatenating and truncating at 25 loses the second source")
    order, counts, _ = M._union_scaffolds([("pc", B_HITS[:1] * 25), ("eng", A_HITS[:1])], [25, 40])
    check(len(order) == 2, f"per-source harvesting keeps both, got {len(order)}")
    check(counts == {"pc": 1, "eng": 1}, f"per-source counts, got {counts}")


def test_union_scaffolds():
    print("test_union_scaffolds")
    order, counts, owner = M._union_scaffolds([("pc", A_HITS), ("eng", B_HITS + A_HITS[:1])], [25, 40])
    keys = [k for k, _, _ in order]
    check(len(keys) == len(set(keys)), "the union is deduplicated")
    check(NAPH in keys, "a scaffold only the second source has is harvested")
    check(owner[NAPH] == "eng", "it is owned by the second source")
    shared = M.harvest_scaffolds(A_HITS[:1], 25)
    check(all(owner[k] == "pc" for k in shared), "a scaffold both sources have is owned by the first")
    check(counts == {"pc": 2, "eng": 3}, f"per-source counts, got {counts}")


def test_round_robin_order():
    """Walk order is interleaved, because mutate_sources returns early at cap or at the deadline."""
    print("test_round_robin_order")
    order, _, _ = M._union_scaffolds([("pc", A_HITS), ("eng", B_HITS)], [25, 40])
    owners = [o for _, _, o in order]
    check(owners == ["pc", "eng", "pc", "eng"], f"sources interleave, got {owners}")
    check(owners.index("eng") == 1, "an eng scaffold is reached second, not after every pc one")
    flat, _, _ = M._union_scaffolds([("all", A_HITS + B_HITS)], [65])
    check(len(flat) == len(order), "a flat harvest finds the same scaffolds, only later")


def test_union_formulas():
    print("test_union_formulas")
    many = ["C", "CC", "CCC", "CCCC", "CCCCC"] * 3          # 5 formulas, 3 votes each
    one = ["c1ccccc1"]                                      # 1 formula, 1 vote
    f = M._union_formulas([("pc", many), ("eng", one)], [25, 40], 5)
    check(M.vec_of_smiles("c1ccccc1") in f, "the second source's own formula survives a popular vote")
    check(len(f) == 5, f"and max_formulas is still respected, got {len(f)}")
    g = M._union_formulas([("pc", many), ("eng", one)], [25, 40], 6)
    check(len(g) == 6 and g[-1] == M.vec_of_smiles("c1ccccc1"), "with room, it is simply appended last")


def test_two_sources():
    print("test_two_sources")
    R = M.build_reach(maxops=3)
    a = ["Oc1ccc2c(c1)oc(-c1ccc(O)cc1)c(O)c2=O"]
    b = a + ["O=c1c(O)c(-c2ccccc2)oc2cc(O)ccc12"]
    one = set(M.mutate(a, R))
    two, prov = M.mutate_sources([("pc", a), ("eng", b)], R)
    check(len(two) < M.MAX_PRODUCTS, "the cap is not reached, so the comparison below is meaningful")
    check(one <= set(two), f"two sources produce a superset ({len(one)} -> {len(two)})")
    check(set(prov.values()) <= {"pc", "eng"}, "provenance labels are the source names")
    check(set(prov) == set(two), "every candidate is attributed")


def test_deadline():
    print("test_deadline")
    R = M.build_reach(maxops=2)
    st = collections.Counter()
    out, prov = M.mutate_sources([("pc", A_HITS)], R, deadline=time.monotonic() - 1, stats=st)
    check(out == [] and prov == {}, "an expired deadline returns empty without raising")
    check(st["hit_deadline"] == 1, "and is counted once")
    st2 = collections.Counter()
    ok, _ = M.mutate_sources([("pc", A_HITS)], R, deadline=time.monotonic() + 600, stats=st2)
    check(st2["hit_deadline"] == 0, "a deadline far in the future never fires")


def test_formula_opt_out():
    """A source at a different mass must donate scaffolds but not target formulas."""
    print("test_formula_opt_out")
    both = M._union_formulas([("pc", A_HITS), ("ana", B_HITS)], [25, 20], 6)
    pc_only = M._union_formulas([("pc", A_HITS), ("ana", B_HITS)], [25, 20], 6, [True, False])
    want = M._union_formulas([("pc", A_HITS)], [25], 6)
    check(pc_only == want, "opting a source out gives exactly the formulas of the remaining ones")
    check(len(pc_only) < len(both), f"and drops its formulas ({len(both)} -> {len(pc_only)})")
    for smi in B_HITS:
        check(M.vec_of_smiles(smi) not in pc_only,
              "an opted-out source's own formula never becomes a target")
    # the scaffolds, however, must still be there -- that is the whole point of the opt-out
    order, counts, _ = M._union_scaffolds([("pc", A_HITS), ("ana", B_HITS)], [25, 20])
    check(counts["ana"] == len(M.harvest_scaffolds(B_HITS, 20)) and counts["ana"] > 0,
          "while its scaffolds are harvested in full")


def test_max_steps():
    print("test_max_steps")
    R = M.build_reach(maxops=2)
    free = collections.Counter()
    o_free, _ = M.mutate_sources([("pc", SHALLOW)], R, stats=free)
    capped = collections.Counter()
    o_cap, _ = M.mutate_sources([("pc", SHALLOW)], R, stats=capped, max_steps=1)
    check(len(o_free) > 0 and len(o_cap) > 0,
          f"both runs produce candidates, so the comparison is real ({len(o_free)}, {len(o_cap)})")
    check(set(o_cap) <= set(o_free), "capping depth can only remove products, never add any")
    check(len(o_cap) < len(o_free), f"and it does remove some ({len(o_free)} -> {len(o_cap)})")
    check(capped["pairs_too_deep"] == 1, "the one 2-step pair it refused is counted separately")
    check(capped["pairs_reachable"] == 1, "leaving exactly the 1-step pair")
    check(free["pairs_too_deep"] == 0, "and an uncapped run never counts one")

    # a cap at the table depth is a no-op, which pins that the filter is on length not identity
    R1 = M.build_reach(maxops=1)
    a = collections.Counter()
    o_a, _ = M.mutate_sources([("pc", SHALLOW)], R1, stats=a)
    b = collections.Counter()
    o_b, _ = M.mutate_sources([("pc", SHALLOW)], R1, stats=b, max_steps=1)
    check(o_a == o_b and len(o_a) > 0, "capping at the table own depth changes nothing")
    check(b["pairs_too_deep"] == 0, "and refuses no pair")


def test_max_steps_per_source():
    print("test_max_steps_per_source")
    R = M.build_reach(maxops=2)
    deep = collections.Counter()
    o_deep, _ = M.mutate_sources([("pc", SHALLOW), ("ana", AZ_HITS)], R, per_source=(25, 20),
                                 stats=deep, contributes_formulas=[True, False])
    st = collections.Counter()
    out, prov = M.mutate_sources([("pc", SHALLOW), ("ana", AZ_HITS)], R, per_source=(25, 20),
                                 stats=st, contributes_formulas=[True, False],
                                 max_steps={"ana": 1})
    check(len(out) > 0, f"a mixed-depth two-source run still produces candidates ({len(out)})")
    check(set(prov.values()) == {"pc", "ana"}, "and both sources are represented")
    check(len(out) < len(o_deep),
          f"capping only `ana` removes only its deep products ({len(o_deep)} -> {len(out)})")
    check(st["pairs_too_deep"] == 1 and deep["pairs_too_deep"] == 0,
          "exactly the one capped source 2-step pair is refused")
    check(sum(1 for v in prov.values() if v == "pc") > 0,
          "the uncapped source keeps its deep products")


def test_three_sources_is_additive():
    print("test_three_sources_is_additive")
    R = M.build_reach(maxops=2)
    two, _ = M.mutate_sources([("pc", SHALLOW), ("eng", B_HITS)], R, per_source=(25, 40))
    three, prov = M.mutate_sources([("pc", SHALLOW), ("eng", B_HITS), ("ana", AZ_HITS)], R,
                                   per_source=(25, 40, 20),
                                   contributes_formulas=[True, True, False])
    check(set(two) <= set(three), "adding a third source never removes a candidate")
    check(len(three) > len(two),
          f"and here it genuinely grows the pool ({len(two)} -> {len(three)})")
    check(sum(1 for v in prov.values() if v == "ana") > 0,
          "the analog source contributes products of its own")
    # the point of the opt-out: every product must sit at a formula the *retrieval* sources
    # proposed, never at one the analog dragged in from its own (different) mass window
    targets = set(M.harvest_formulas(SHALLOW, 25)) | set(M.harvest_formulas(B_HITS, 40))
    check(all(M.vec_of_smiles(x) in targets for x in three),
          "and every product sits at a target formula taken from the retrieval sources only")


if __name__ == "__main__":
    for fn in [test_palette_vectors_match_smarts, test_reach_table, test_elem_vec_and_formula,
               test_harvest, test_grow_formula_exact, test_socket_order, test_mutate_end_to_end,
               test_single_source_equivalence, test_per_source_limit_trap, test_union_scaffolds,
               test_round_robin_order, test_union_formulas, test_two_sources, test_deadline,
               test_formula_opt_out, test_max_steps, test_max_steps_per_source,
               test_three_sources_is_additive]:
        fn()
    print()
    if FAIL:
        print(f"{len(FAIL)} FAILED")
        for f in FAIL:
            print("  -", f)
        sys.exit(1)
    print("all tests passed")
