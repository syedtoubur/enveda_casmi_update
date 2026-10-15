"""c3-gen-union (campaign 20261015-c3-gen-slot2, K1) fixture tests: one scale for beam leaves and mutator edits.

    python build/test_novel_p4.py

Synthetic, CPU, no Drive. Reuses the tiny decoder of test_novel_p3.
"""
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("NOVEL_ALLOW_ANY_EMB", "1")

import torch  # noqa: E402

import novel_decoder as nd  # noqa: E402
import novel_p4 as p4  # noqa: E402
from test_novel_p3 import SMILES, TABLE, VOCAB, tiny  # noqa: E402


def memory_for(model, formula, k=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    emb = torch.randn(k, model.emb_proj.in_features, generator=g)
    ad = torch.zeros(k, dtype=torch.long)
    ce = torch.full((k,), nd.CE_UNKNOWN, dtype=torch.long)
    fv = torch.as_tensor(nd.formula_vector(nd.parse_formula(formula)), dtype=torch.float32)
    return model.build_memory(emb, ad, ce, fv)


def fstr(counts):
    return "".join(f"{e}{c if c > 1 else ''}" for e, c in sorted(counts.items()))


def test_teacher_logp_equals_beam_logp():
    """teacher_logp == the sum of `_logprobs` (the beam's own step scores) along the same token path."""
    model = tiny().eval()
    for smi in SMILES[:3]:
        _, can, s, form = nd.prep_structure(smi)
        f = fstr(form)
        mem, pad = memory_for(model, f)
        _counts, t, ids = nd._prep_search(model, mem, pad, f, TABLE)
        total = 0.0
        for tok in VOCAB.encode(s) + [model.eos_id]:
            lp, legal, dead = nd._logprobs(model, ids, mem, pad, [t])
            assert bool(legal[0, tok]), (smi, tok)
            total += float(lp[0, tok])
            if tok != model.eos_id:
                ids = torch.cat([ids, torch.tensor([[tok]])], dim=1)
                t = t.push(tok)
        tl = p4.teacher_logp(model, mem, pad, TABLE, s, f, torch)
        assert tl is not None and abs(tl - total) < 1e-3 * max(1.0, abs(total)), (smi, tl, total)
        assert p4.to_selfies(smi)[0] == s and p4.to_selfies(smi)[1] == form


def test_teacher_logp_rejects_wrong_formula():
    model = tiny().eval()
    _, can, s, form = nd.prep_structure(SMILES[0])
    other = dict(form)
    other["C"] = other["C"] + 2
    f2 = fstr(other)
    mem, pad = memory_for(model, f2)
    assert p4.teacher_logp(model, mem, pad, TABLE, s, f2, torch) is None
    assert p4.teacher_logp(model, mem, pad, TABLE, "[Xx]", f2, torch) is None


def test_predicted_formulas_rank_order_and_vocab():
    forms = ["C10H12O3|gen", "C10H12O3", "C9H8O4", "C12H24ClO", None, "garbage!", "C8H10N4O2", "C7H6O2"]
    allowed = set(nd.ALLOWED_ELEMENTS) - {"Cl"}
    assert p4.predicted_formulas(forms, k=3, allowed=allowed) == ["C10H12O3", "C9H8O4", "C8H10N4O2"]
    assert p4.predicted_formulas(forms, k=2) == ["C10H12O3", "C9H8O4"]
    assert p4.predicted_formulas([], k=3) == []


def test_union_rank_dedupes_and_merges_sources():
    key = {"A1": "K1", "A2": "K1", "B": "K2", "C": "K3"}.get
    out = p4.union_rank([("dec", [("A1", -3.0), ("B", -5.0)]), ("mut", [("A2", -2.0), ("C", None), ("B", -9.0)])], key)
    assert [(r[0], r[1], r[2], r[3]) for r in out] == [("A2", "K1", -2.0, "dec+mut"), ("B", "K2", -5.0, "dec+mut")]
    assert p4.rank_of(out, "K2") == 2 and p4.rank_of(out, "K9") == 0


def test_k1_report():
    rows = ([dict(rank=1, acyclic=False, timstof=False, has_formula=True, src_top1="dec")] * 2 +
            [dict(rank=3, acyclic=True, timstof=True, has_formula=True)] +
            [dict(rank=0, acyclic=True, timstof=False, has_formula=False)] * 97)
    r = p4.k1_report(rows)
    assert abs(r["all"]["h"] - 0.02) < 1e-9 and r["K1"] == "DEAD"
    assert abs(r["all"]["hit@3"] - 0.03) < 1e-9 and r["ring"]["n"] == 2 and r["ring"]["h"] == 1.0
    assert abs(r["formula_recall"] - 0.03) < 1e-9 and r["src_of_top1_hits"] == {"dec": 2}
    rows[3:6] = [dict(rank=1, acyclic=True, timstof=False, has_formula=True)] * 3
    assert p4.k1_report(rows)["K1"] == "PASS"


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    bad = 0
    for n, f in tests:
        try:
            f(); print("ok  ", n)
        except Exception as e:
            bad += 1; print("FAIL", n, repr(e))
    print(f"{len(tests) - bad}/{len(tests)} passed")
    sys.exit(1 if bad else 0)
