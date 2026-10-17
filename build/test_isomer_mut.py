"""Tests for build/isomer_mut.py (c3-iso I1). Plain functions; run: python build/test_isomer_mut.py"""
import os
import pickle
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import isomer_mut as im  # noqa: E402

DUMP = "E:/ENVEDA_casmi/research/harness/dumps/pseudo_c3/post_dump/state.pkl"
FAIL = []


def check(c, msg):
    print(("  ok   " if c else "  FAIL ") + msg)
    if not c:
        FAIL.append(msg)


_PARENTS = None


def real_parents(n=20):
    """Real natural-product parents: the base-list top candidates of pseudo_c3 dump molecules (no truth involved)."""
    global _PARENTS
    if _PARENTS is None:
        st = pickle.load(open(DUMP, "rb"))
        out = []
        for m in st["te_mids"]:
            b = st["base"].get(m)
            lst = list(b[0]) if isinstance(b, tuple) else []
            if lst:
                out.append(lst[0])
            if len(out) >= n:
                break
        _PARENTS = out
    return _PARENTS


def test_formula_preserved_real_parents():
    print("test_formula_preserved_real_parents")
    st = {}
    bad = nchild = 0
    for p in real_parents():
        f0 = im.formula(im._parse(p))
        for s, op, n in im.mutate(p, cap=150, deadline=time.monotonic() + 30, stats=st):
            nchild += 1
            if im.formula(im.Chem.MolFromSmiles(s)) != f0:
                bad += 1
    check(nchild > 0 and bad == 0, f"{nchild} children of {len(real_parents())} real parents, formula drift {bad}")
    print("   internal formula drops (edits rejected before return):", st.get("formula_drop", 0))


def test_no_child_equals_parent():
    print("test_no_child_equals_parent")
    bad = same = 0
    for p in real_parents():
        k0 = im.ik14_plain(p)
        bad += sum(k == k0 for _, k, _, _, _ in im.mutate_many([p], cap=100, deadline=time.monotonic() + 30))
        raw = im.mutate(p, cap=100, deadline=time.monotonic() + 30)
        same += sum(im.ik14_plain(s) == k0 for s, _, _ in raw)
    check(bad == 0, f"mutate_many children equal to parent IK14: {bad} (raw mutate had {same} InChI-tautomer-equal, filtered)")


def test_nonvacuous_mean_children():
    print("test_nonvacuous_mean_children")
    ns = [len(im.mutate(p, cap=300, deadline=time.monotonic() + 30)) for p in real_parents(20)]
    check(sum(ns) / len(ns) > 0, f"mean children over {len(ns)} real molecules = {sum(ns) / len(ns):.1f}")


def reach(parent, target, max_moves=1):
    k = im.ik14_plain(target)
    return [op for s, op, n in im.mutate(parent, max_moves=max_moves, cap=0) if im.ik14_plain(s) == k]


def test_known_pairs_one_move():
    print("test_known_pairs_one_move")
    r = reach("COc1ccc(C=CC(=O)O)cc1", "COc1cccc(C=CC(=O)O)c1")
    check(bool(r), f"4-OMe -> 3-OMe cinnamic acid in 1 move via {r}")
    r = reach("OC[C@H]1O[C@@H](OC2=C(c3ccc(O)c(O)c3)Oc3cc(O)cc(O)c3C2=O)[C@H](O)[C@@H](O)[C@@H]1O",
              "OC[C@H]1O[C@@H](Oc2cc(O)c3c(c2)OC(c2ccc(O)c(O)c2)=C(O)C3=O)[C@H](O)[C@@H](O)[C@@H]1O")
    check(bool(r), f"quercetin 3-O-glc -> 7-O-glc in 1 move via {r}")


def test_each_operator_fires():
    print("test_each_operator_fires")
    cases = {
        "subst_shift": "COc1ccc(C=CC(=O)O)cc1",
        "methyl_swap": "COc1ccc(C=CC(=O)O)cc1",
        "group_move": "COc1ccc(O)cc1C(=O)O",
        "db_shift": "CC(C)=CCc1ccccc1",
        "chain_iso": "CCCCCCO",
    }
    for op, smi in cases.items():
        ops = {o.split("+")[0] for _, o, _ in im.mutate(smi, max_moves=1, cap=0)}
        check(op in ops, f"{op} fires on {smi} (got {sorted(ops)})")
    # a methyl-swap in the reverse direction
    ops = {o for _, o, _ in im.mutate("OCc1ccccc1C(=O)O", max_moves=1, cap=0)}
    check("methyl_swap" in ops, "methyl_swap reverse (CH2OH -> OMe)")


def test_two_moves_and_cap_deadline():
    print("test_two_moves_and_cap_deadline")
    p = real_parents()[0]
    ch = im.mutate_many([p], cap=50, deadline=time.monotonic() + 30)
    check(len(ch) <= 50, f"cap respected ({len(ch)} <= 50)")
    ch = im.mutate("COc1ccc(C=CC(=O)O)cc1", max_moves=2, cap=0)
    check(any(n == 2 for _, _, n in ch), "two-move children exist")
    t = time.monotonic()
    im.mutate(real_parents()[1], deadline=time.monotonic() - 1)
    check(time.monotonic() - t < 5, "expired deadline returns promptly")


def test_generate_phase_never_touches_truth():
    """A1: the frozen lists are written before any truth is read (static check of isomer_reach.generate/_work)."""
    print("test_generate_phase_never_touches_truth")
    import ast
    src = open(os.path.join(HERE, "isomer_reach.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    bad = []
    for fn in tree.body:
        if isinstance(fn, ast.FunctionDef) and fn.name in ("generate", "_work", "spectra_pseudo_c3", "spectra_val", "_init", "generate_i1b", "_compact", "settings_of"):
            for n in ast.walk(fn):
                nm = n.id if isinstance(n, ast.Name) else n.attr if isinstance(n, ast.Attribute) else ""
                if nm in ("truths", "T") or "truth" in nm.lower():
                    bad.append((fn.name, nm))
                if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value in ("inchikey14",):
                    bad.append((fn.name, n.value))
    check(not bad, f"no truth access in generate/_work/spectra loaders: {bad}")


def test_inner_deadline():
    """I1b: the deadline is enforced inside generation (per move / per child), not only between parents."""
    print("test_inner_deadline")
    ps = real_parents(6)
    for budget in (0.5, 2.0):
        t = time.monotonic()
        st = {}
        ch = im.mutate_many(ps, cap=600, deadline=time.monotonic() + budget, stats=st)
        el = time.monotonic() - t
        check(el < budget + 8.0 and len(ch) > 0, f"budget {budget}s: returned after {el:.2f}s with {len(ch)} children, deadline_hit={st.get('deadline_hit')}")
    t = time.monotonic()
    st = {}
    ch = im.mutate_many(ps, cap=600, deadline=time.monotonic() - 1, stats=st)
    check(st.get("deadline_hit") == 1 and len(ch) == 0, "expired deadline: no generation, flagged")
    # unbounded run produces strictly more than a 0.5 s run (the deadline really cuts)
    full = im.mutate_many(ps[:2], cap=600, deadline=time.monotonic() + 60)
    cut = im.mutate_many(ps[:2], cap=600, deadline=time.monotonic() + 0.3)
    check(len(cut) < len(full), f"0.3 s run {len(cut)} < unbounded {len(full)} children")


def test_freeze_before_truth():
    """A1 (I1b.0): run_i1b calls generate_i1b before measure_i1b; measure_i1b refuses a missing frozen file;
    generate_i1b/_compact never name truths."""
    print("test_freeze_before_truth")
    import ast
    import isomer_reach as ir
    tree = ast.parse(open(os.path.join(HERE, "isomer_reach.py"), encoding="utf-8").read())
    fn = {f.name: f for f in tree.body if isinstance(f, ast.FunctionDef)}
    calls = [(n.lineno, n.func.id) for n in ast.walk(fn["run_i1b"]) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    g = min(l for l, nm in calls if nm == "generate_i1b")
    m = min(l for l, nm in calls if nm == "measure_i1b")
    check(g < m, f"run_i1b: generate_i1b (line {g}) precedes measure_i1b (line {m})")
    bad = [n.id for f in ("generate_i1b", "_compact") for n in ast.walk(fn[f]) if isinstance(n, ast.Name) and "truth" in n.id.lower()]
    check(not bad, f"no truth names in generate_i1b/_compact: {bad}")
    try:
        ir.measure_i1b("E:/casmi/c3_iso/i1b_does_not_exist.pkl")
        check(False, "measure_i1b on a missing frozen file must raise")
    except AssertionError as e:
        check("freeze-before-truth" in str(e), "measure_i1b refuses a missing frozen file")


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_") and callable(f):
            f()
    print("FAILED: %s" % FAIL if FAIL else "ALL OK")
    sys.exit(1 if FAIL else 0)
