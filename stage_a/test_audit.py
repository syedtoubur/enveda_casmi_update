import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "build"))
import a4, audit
import iso_rank as ir

FRAG_CFG = dict(ISO_M_ICE=0.0, ISO_LISTS=audit.CPU_LISTS)


def K(prefix, i):
    return (prefix + chr(65 + i // 26) + chr(65 + i % 26)).ljust(14, "Z")


def record(kids, row_n=25, pfrag=0.3):
    """Hand-made A3 record. kids: [(name, parent_index list, frag, nl)]; SMILES are labels resolved by the keymap."""
    row = [f"p{i}" for i in range(row_n)]
    km = {s: (K("PAR", i), K("PAR", i)) for i, s in enumerate(row)}
    ks = []
    for j, (name, pis, frag, nl) in enumerate(kids):
        km.setdefault(name, (K("KID", j), K("KID", j)))
        ks.append(dict(smiles=name, ik14=km[name][0], op="subst_shift", frag=frag, nl=nl,
                       pairs=[(pi, f"op{t}") for t, pi in enumerate(pis)]))
    parents = [dict(smiles=row[i], ik14=km[row[i]][0], rank=i + 1, frag=pfrag) for i in range(10)]
    return dict(mid="M", row=row, parents=parents, kids=ks, keymap=km)


def patch(*recs):
    a4.patch_keys({k: v for r in recs for k, v in r["keymap"].items()})


def test_top_child_is_the_child_apply_inserts():
    r = record([("ca", [2, 2, 3], 0.9, 0.5), ("cb", [0], 0.1, 0.0), ("cc", [5], 0.2, 0.1)])
    patch(r)
    c, pos = audit.top_child(r)
    assert c is not None and c["smiles"] == "ca" and pos == 3 and c["multiplicity"] == 3
    row, rec = a4.build_rec(r, frag_only=True)
    new, info = ir.apply(row, rec, dict(FRAG_CFG, ISO_TAIL=False))
    assert info["fired"] and new[pos] == "ca"                     # directly below p (0-based index pos)


def test_child_already_in_row_is_skipped():
    r = record([("p7", [1, 2], 0.9, 0.9), ("cb", [4], 0.5, 0.2)])     # best child is a row member
    patch(r)
    c, pos = audit.top_child(r)
    assert c["smiles"] == "cb" and pos == 5


def test_dnl_sign_rules_and_ties():
    r = record([("ca", [0, 1], 0.9, 0.4)])
    patch(r)
    pk = r["parents"][0]["ik14"]
    up = audit.signals(r, {pk: 0.1})
    down = audit.signals(r, {pk: 0.6})
    assert up["has"] and abs(up["dnl"] - 0.3) < 1e-12 and abs(up["dfrag"] - 0.6) < 1e-12 and up["mult"] == 2
    assert audit.fires(up)["S_nl"] and audit.fires(up)["rule_A"] and not audit.fires(up)["rule_B"]
    assert not audit.fires(down)["S_nl"] and audit.fires(down)["S_mult"]
    t = record([("cz", [0], 0.9, 0.0)])
    patch(t)
    tie = audit.signals(t, {t["parents"][0]["ik14"]: 0.0})
    s = audit.summarize([tie, up])
    assert s["n"] == 2 and s["tie_both0"] == 0.5 and s["tie_dnl0"] == 0.5
    assert s["rules"]["S_nl"]["fire"] == 0.5 and s["rules"]["S_nl"]["p_pos1_given_fire"] == 1.0


def test_hit_counts_only_c_star_and_missing_parent_nl():
    r = record([("ca", [0], 0.9, 0.4), ("cb", [1], 0.1, 0.0)])
    patch(r)
    truth = r["keymap"]["ca"][1]
    s = audit.signals(r, {}, truth)
    assert s["hit"] and s["dnl"] is None and not audit.fires(s)["S_nl"]
    assert not audit.signals(r, {}, r["keymap"]["cb"][1])["hit"]
    summ = audit.summarize([s])
    assert summ["h_top"] == 1.0 and summ["nl_missing"] == 1.0


def test_no_children_does_not_fire():
    r = record([])
    patch(r)
    s = audit.signals(r, {})
    assert not s["has"] and not any(audit.fires(s).values())
    assert audit.summarize([s])["rules"]["S_mult"]["fire"] == 0.0


def test_verdict_arithmetic():
    v = audit.verdict(0.04, 0.10)
    assert abs(v["net"] - (0.025 - 0.75 * 0.367 * 0.04)) < 1e-12 and v["alive"]
    assert not audit.verdict(0.05, 0.5)["alive"]                # FPR bar binds
    assert not audit.verdict(0.0, 0.01)["alive"]                # noise floor binds


def test_fired_truth_diagnostic():
    r = record([("ca", [2], 0.9, 0.5)])
    patch(r)
    assert audit.fired_truth(r, r["keymap"]["ca"][1], FRAG_CFG, frag_only=True) == (True, True)
    assert audit.fired_truth(r, "NOTTHETRUTHXYZ", FRAG_CFG, frag_only=True) == (True, False)
    lo = record([("ca", [2], 0.1, 0.5)])                         # frag(c) < frag(p): no fire
    patch(lo)
    assert audit.fired_truth(lo, None, FRAG_CFG, frag_only=True) == (False, False)
