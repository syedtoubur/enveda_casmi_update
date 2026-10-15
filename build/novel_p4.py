"""Phase 1 of campaign 20261015-c3-gen-slot2 (c3-gen-union, gate K1): the Class-3 generator union.

Two generators, one leak-free scale (gate_report 2026-10-15 amendment: no FPNet in the novel channel):
  * decoder  - `novel_decoder.search_masked_beam` per PREDICTED formula (top-k distinct formulas of the molecule's
               retrieved candidates in the +1.45 ppm-centred window; never the oracle formula);
  * mutator  - c3v2 analog edits of retrieved anchors (`delta_m_mutator`), scored by `teacher_logp`, the decoder's
               masked teacher-forced log p(SELFIES | spectra, formula). For a beam leaf the two are the same number
               (test), so the union is ranked on one scale.
Union = IK14-deduped (best score kept, sources merged), ranked by score. K1 metrics: top-1 exact rate h, hit@1-4,
split acyclic / ring-bearing and timsTOF.

Pure functions only; the Colab driver is build/novel_p4_colab.py.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np

import novel_decoder as nd

FORMULA_K = 3                      # predicted formulas per molecule
PPM_CENTER, PPM_WIN = 1.45, 10.0   # CLAUDE.md: timsTOF offset; the engines' own window


# --------------------------------------------------------------------------------------------- #
# Predicted formulas (no oracle): the formulas the retrieval channels already put at the query mass
# --------------------------------------------------------------------------------------------- #

def _plain_formula(f):
    """'C10H12O3|gen' / 'C10H12O3' -> 'C10H12O3'; anything unparsable -> None."""
    if not isinstance(f, str):
        return None
    f = f.split("|", 1)[0].strip()
    return f if re.fullmatch(r"(?:[A-Z][a-z]?\d*)+", f) else None


def predicted_formulas(ranked_formulas, k=FORMULA_K, allowed=None):
    """Distinct formulas in rank order of the molecule's retrieved candidates, first k.

    ranked_formulas: formulas of the fused list in rank order (BASE forms, then ENG / PC formulas).
    Rank order, not frequency: the first channel hit is the best evidence the pipeline has, and a frequency vote
    would favour formulas with many isomers in the pool. Formulas with elements outside the decoder's vocabulary
    (`allowed`) are skipped, since the search cannot produce them."""
    out = []
    for f in ranked_formulas:
        p = _plain_formula(f)
        if p is None or p in out:
            continue
        if allowed is not None and not set(nd.parse_formula(p)) <= set(allowed):
            continue
        out.append(p)
        if len(out) >= k:
            break
    return out


# --------------------------------------------------------------------------------------------- #
# Teacher-forced masked log p: the decoder's score of ANY structure, on the beam's own scale
# --------------------------------------------------------------------------------------------- #

def teacher_logp(model, memory, mem_pad, table, selfies_str, formula, torch):
    """Masked cumulative log p of `selfies_str` (+ </s>) under formula-conditioned `memory`.

    Same masks and log-softmax as `search_masked_beam`, so for a beam leaf this equals the leaf's logp.
    -> float, or None when the structure cannot be derived inside `formula` (a token is illegal, or </s> is not
    legal at the end), i.e. the beam could never have produced it."""
    vocab = table.vocab
    try:
        ids = vocab.encode(selfies_str)
    except (KeyError, ValueError):
        return None
    if not ids:
        return None
    counts = nd.parse_formula(formula) if isinstance(formula, str) else dict(formula)
    t = nd.Tracker(table, {e: int(c) for e, c in counts.items() if c})
    if t.missing:
        return None
    seq = [model.sos_id] + list(ids)
    tok = torch.tensor([seq], dtype=torch.long, device=memory.device)
    with torch.no_grad():
        logits = model.forward(tok, memory, mem_pad)[0].float()       # (L, V): position i predicts seq[i + 1]
    total = 0.0
    for i, nxt in enumerate(list(ids) + [model.eos_id]):
        legal = t.legal_mask()
        if not legal[nxt]:
            return None
        row = logits[i].masked_fill(~torch.from_numpy(legal).to(logits.device), nd.NEG_INF)
        total += float(torch.log_softmax(row, dim=-1)[nxt])
        if nxt != model.eos_id:
            t = t.push(nxt)
    return total


def to_selfies(smiles):
    """SMILES -> (SELFIES, formula) with the decoder's own training preprocessing (`prep_structure`), or None."""
    st, _can, s, form = nd.prep_structure(smiles) if isinstance(smiles, str) else ("parse", None, None, None)
    return (s, form) if st == "ok" else None


# --------------------------------------------------------------------------------------------- #
# Union + K1 metrics
# --------------------------------------------------------------------------------------------- #

def union_rank(lists, key_fn):
    """lists: iterable of (src, [(smiles, score)]). -> [(smiles, ik, score, 'src1+src2')] best score first.

    IK14-deduped by `key_fn` (the metric's tautomer-canonical IK14); a structure found by several generators keeps
    its best score and every source tag."""
    best, srcs = {}, {}
    for src, items in lists:
        for smi, sc in items:
            if sc is None or not np.isfinite(sc):
                continue
            k = key_fn(smi)
            if not k:
                continue
            srcs.setdefault(k, set()).add(src)
            if k not in best or sc > best[k][1]:
                best[k] = (smi, sc)
    out = [(s, k, sc, "+".join(sorted(srcs[k]))) for k, (s, sc) in best.items()]
    return sorted(out, key=lambda r: -r[2])


def rank_of(ranked, truth_ik, depth=25):
    for i, r in enumerate(ranked[:depth], 1):
        if r[1] == truth_ik:
            return i
    return 0


def k1_report(rows):
    """rows: dicts with rank (0 = miss), acyclic, timstof, has_formula (truth formula among the predicted ones).

    -> h (top-1 exact rate over ALL molecules), hit@1..4, @25, MRR@25, splits, and the formula-recall ceiling."""
    n = len(rows)
    r = np.array([x["rank"] for x in rows])

    def block(mask):
        m = int(mask.sum())
        rr = r[mask]
        return dict(n=m, h=float((rr == 1).mean()) if m else float("nan"),
                    **{f"hit@{k}": float(((rr >= 1) & (rr <= k)).mean()) if m else float("nan") for k in (2, 3, 4, 25)},
                    mrr25=float(np.where(rr > 0, 1.0 / np.maximum(rr, 1), 0.0).mean()) if m else float("nan"))
    acyc = np.array([bool(x["acyclic"]) for x in rows])
    tims = np.array([bool(x["timstof"]) for x in rows])
    form = np.array([bool(x["has_formula"]) for x in rows])
    out = dict(all=block(np.ones(n, bool)), acyclic=block(acyc), ring=block(~acyc), timstof=block(tims),
               formula_recall=float(form.mean()) if n else float("nan"),
               given_formula=block(form))
    out["src_of_top1_hits"] = dict(Counter(x.get("src_top1", "") for x in rows if x["rank"] == 1))
    out["K1"] = "PASS" if out["all"]["h"] >= 0.03 else "DEAD"
    return out
