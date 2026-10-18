"""c3 signal audit (gate_report 2026-10-18 "c3 signal audit, PRE-REGISTERED"). Pure functions over the Stage A A3 records.

c* = the RRF-best child not in the V22 row, chosen exactly as iso_rank.apply does (CPU lists only); p = its best-ranked
parent. Signals dnl, mult, dfrag compare c* with p. Parent nl is not in the A3 records: the caller computes it with
iso_nl.nl_scores on the same spectra and passes {parent ik14: nl}. Truth only scores hits; it never selects c*."""
from __future__ import annotations

import numpy as np

import a4
import iso_rank as ir

CPU_LISTS = ("mult", "parent_rank", "frag", "nl")
NL_T = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2)
MULT_K = (1, 2, 3, 4, 5, 6)
FRAG_T = (0.0, 0.01, 0.02, 0.05, 0.1)
FPR_BAR, PI, COST, NOISE = 0.045, 0.25, 0.367, 0.004


def _gt(v, t):
    return v is not None and v > t


RULES = {
    "S_nl": lambda s: _gt(s["dnl"], 0.0),
    "S_mult": lambda s: s["mult"] is not None and s["mult"] >= 2,
    "S_frag": lambda s: _gt(s["dfrag"], 0.0),
    "rule_A": lambda s: _gt(s["dnl"], 0.0) and s["mult"] >= 2,
    "rule_B": lambda s: s["mult"] is not None and s["mult"] >= 3 and _gt(s["dfrag"], 0.0),
}


def top_child(r, lists=CPU_LISTS, k=10):
    """-> (c* parent_map record, 1-based pos(p) or None); (None, None) if no child is outside the row."""
    row, rec = a4.build_rec(r)
    recs = ir.parent_map(rec["children"])
    if not row or not recs:
        return None, None
    order = ir.rrf(recs, rec["scores"], k=k, lists=tuple(lists))
    row_keys = [ir._keys(e) for e in row]

    def child_keys(ik):
        return set(ir._keys(recs[ik]["smiles"])) | {ik}

    cand = next((ik for ik in order if not any(child_keys(ik) & rk for rk in row_keys)), None)
    if cand is None:
        return None, None
    c = recs[cand]
    pos = next((i for i, rk in enumerate(row_keys) if c["parent_ik14"] in rk), None)
    return c, (pos + 1 if pos is not None else None)


def _diff(a, b):
    return None if a is None or b is None else float(a) - float(b)


def signals(r, parent_nl, truth_taut=None, lists=CPU_LISTS):
    """One molecule -> dict(has, pos, mult, dnl, dfrag, nl_c, nl_p, hit). has = c* exists AND its parent is in the row."""
    c, pos = top_child(r, lists)
    out = dict(has=False, pos=None, mult=None, dnl=None, dfrag=None, nl_c=None, nl_p=None, hit=False)
    if c is None or pos is None:
        return out
    kd = next(k for k in r["kids"] if k["ik14"] == c["ik14"])
    pf = next((p for p in r["parents"] if p["ik14"] == c["parent_ik14"]), {})
    nl_c, nl_p = kd.get("nl"), parent_nl.get(c["parent_ik14"])
    hit = bool(truth_taut) and (r["keymap"].get(c["smiles"]) or (None, None))[1] == truth_taut
    out.update(has=True, pos=pos, mult=int(c["multiplicity"]), nl_c=nl_c, nl_p=nl_p, dnl=_diff(nl_c, nl_p),
               dfrag=_diff(kd.get("frag"), pf.get("frag")), hit=hit)
    return out


def fires(s):
    return {name: bool(s["has"] and f(s)) for name, f in RULES.items()}


def curves(sigs):
    n = max(len(sigs), 1)
    h = [s for s in sigs if s["has"]]
    return dict(dnl={str(t): sum(_gt(s["dnl"], t) for s in h) / n for t in NL_T},
                mult={str(k): sum(s["mult"] >= k for s in h) / n for k in MULT_K},
                dfrag={str(t): sum(_gt(s["dfrag"], t) for s in h) / n for t in FRAG_T})


def summarize(sigs):
    """Fire rates over ALL molecules (no c* = no fire), ties, P(pos(p) = 1 | fire), hit rates."""
    n = len(sigs)
    if not n:
        return dict(n=0)
    h = [s for s in sigs if s["has"]]
    with_nl = [s for s in h if s["dnl"] is not None]
    fr = [fires(s) for s in sigs]
    out = dict(n=n, has_rate=len(h) / n, h_top=sum(s["hit"] for s in sigs) / n,
               tie_dnl0=(sum(s["dnl"] == 0 for s in with_nl) / len(with_nl)) if with_nl else None,
               tie_both0=(sum(s["nl_c"] == 0 and s["nl_p"] == 0 for s in with_nl) / len(with_nl)) if with_nl else None,
               nl_missing=(len(h) - len(with_nl)) / n, curves=curves(sigs), rules={})
    for name in RULES:
        fi = [i for i, f in enumerate(fr) if f[name]]
        out["rules"][name] = dict(
            fire=len(fi) / n, n_fire=len(fi), tp=sum(sigs[i]["hit"] for i in fi) / n,
            p_pos1_given_fire=(sum(sigs[i]["pos"] == 1 for i in fi) / len(fi)) if fi else None)
    return out


def verdict(fpr, tpr):
    net = PI * tpr - (1 - PI) * COST * fpr
    return dict(fpr=fpr, tpr=tpr, net=net, alive=bool(fpr <= FPR_BAR and net >= NOISE))


def verdicts(c12_pooled, c3_half_a):
    return {name: verdict(c12_pooled["rules"][name]["fire"], c3_half_a["rules"][name]["tp"]) for name in RULES}


def fired_truth(r, truth_taut, cfg, frag_only=False):
    """Diagnostic: (fired, inserted child is the truth) for iso_rank.apply with the tail off."""
    row, rec = a4.build_rec(r, frag_only)
    new, info = ir.apply(row, rec, dict(cfg, ISO_TAIL=False))
    if not info["fired"]:
        return False, False
    ins = next((e for e in new if e not in row), None)
    return True, bool(truth_taut) and ins is not None and (r["keymap"].get(ins) or (None, None))[1] == truth_taut
