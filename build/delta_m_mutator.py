"""Scaffold harvesting + formula-space Delta-M mutation (specs_v4n_update M2).

Pure RDKit + numpy: no `casmi`, no torch, no network. The f.z screen lives in the runner
(`mut_runner.py`) so this module stays testable offline.

Per molecule, over one or more retrieval sources (PubChem top-25, engine-A top-40, ...):
  1. harvest Bemis-Murcko scaffolds from each source separately, then union them (empty scaffold
     -> the whole hit).  Harvesting per source matters: `harvest_scaffolds` truncates its input at
     `limit`, so concatenating a 25-hit and a 40-hit list and passing limit=25 silently throws the
     second source away;
  2. take the distinct molecular formulas of those same hits as the target-formula candidates
     (every source is already inside the +-10 ppm window, so this needs no formula enumeration);
  3. for each (scaffold, formula) pair, Delta = formula - formula(scaffold); look Delta up in the
     precomputed reachable table (exact, formula space -- no ppm tolerance anywhere);
  4. apply the decomposition's transforms through reaction SMARTS, keeping a capped frontier;
  5. keep products whose formula equals the target exactly.

Element order is fixed by EL. Every palette vector was checked against its SMARTS by
`test_delta_m_mutator.py::test_palette_vectors_match_smarts`.
"""
from __future__ import annotations

import collections
import itertools
import time

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

EL = ("C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I")
_EI = {e: i for i, e in enumerate(EL)}

MAXOPS = 6              # measured knee: depth 6 reaches 85.5% of oracle scaffolds, depth 8 only +7.3pp
MAX_DEC = 4             # decompositions kept per delta -- see build_reach
BEAM = 200              # frontier cap per transform step
MAX_PRODUCTS = 8000     # hard cap on products per molecule


def _v(**kw):
    o = [0] * len(EL)
    for k, n in kw.items():
        o[_EI[k]] = n
    return tuple(o)


# name -> (formula delta, [reaction SMARTS])  -- each SMARTS replaces one H unless noted
PALETTE = {
    # --- plan.md palette (9) ---
    "+O":           (_v(O=1),                      ["[cH:1]>>[c:1]O", "[CX4H1,CX4H2,CX4H3:1]>>[C:1]O"]),
    "+CH2":         (_v(C=1, H=2),                 ["[cH:1]>>[c:1]C", "[OX2H:1]>>[O:1]C", "[NX3H1,NX3H2:1]>>[N:1]C"]),
    "+CH2O":        (_v(C=1, H=2, O=1),            ["[cH:1]>>[c:1]OC"]),
    "+CO":          (_v(C=1, O=1),                 ["[cH:1]>>[c:1]C=O", "[NX3H1,NX3H2:1]>>[N:1]C=O"]),
    "+C2H2O":       (_v(C=2, H=2, O=1),            ["[OX2H:1]>>[O:1]C(C)=O", "[NX3H1,NX3H2:1]>>[N:1]C(C)=O"]),
    "+hexose":      (_v(C=6, H=10, O=5),           ["[OX2H:1]>>[O:1]C1OC(CO)C(O)C(O)C1O"]),
    "+deoxyhexose": (_v(C=6, H=10, O=4),           ["[OX2H:1]>>[O:1]C1OC(C)C(O)C(O)C1O"]),
    "+prenyl":      (_v(C=5, H=8),                 ["[cH:1]>>[c:1]CC=C(C)C", "[OX2H:1]>>[O:1]CC=C(C)C"]),
    "-2H":          (_v(H=-2),                     ["[CX4H2:1][CX4H2:2]>>[CH:1]=[CH:2]"]),
    # --- extensions (11): N / S / P / halogen, which GM.0a showed account for 34.1% of misses ---
    "+NH":          (_v(N=1, H=1),                 ["[cH:1]>>[c:1]N", "[CX4H1,CX4H2,CX4H3:1]>>[C:1]N"]),
    "+CONH":        (_v(C=1, H=1, N=1, O=1),       ["[cH:1]>>[c:1]C(N)=O"]),
    "+NHCH2":       (_v(C=1, H=3, N=1),            ["[cH:1]>>[c:1]NC"]),
    "+SH":          (_v(S=1),                      ["[cH:1]>>[c:1]S"]),
    "+SO3":         (_v(S=1, O=3),                 ["[OX2H:1]>>[O:1]S(=O)(=O)O"]),
    "+PO3":         (_v(P=1, O=3, H=1),            ["[OX2H:1]>>[O:1]P(=O)(O)O"]),
    "+Cl":          (_v(Cl=1, H=-1),               ["[cH:1]>>[c:1]Cl"]),
    "+F":           (_v(F=1, H=-1),                ["[cH:1]>>[c:1]F"]),
    "+Br":          (_v(Br=1, H=-1),               ["[cH:1]>>[c:1]Br"]),
    "+I":           (_v(I=1, H=-1),                ["[cH:1]>>[c:1]I"]),
    "+H2O":         (_v(H=2, O=1),                 ["[CX3:1]=[CX3:2]>>[C:1][C:2]O"]),
}

_RXN: dict[str, list] = {}


def _rxns(name):
    if name not in _RXN:
        _RXN[name] = [AllChem.ReactionFromSmarts(s) for s in PALETTE[name][1]]
    return _RXN[name]


# ------------------------------------------------------------------ formula space
def elem_vec(mol):
    """Element-count vector of `mol` including implicit H. None if it uses an element outside EL."""
    c = collections.Counter()
    for a in Chem.AddHs(mol).GetAtoms():
        s = a.GetSymbol()
        if s not in _EI:
            return None
        c[s] += 1
    return tuple(c[e] for e in EL)


def vec_of_smiles(smi):
    m = Chem.MolFromSmiles(smi)
    return None if m is None else elem_vec(m)


def fmt_vec(v):
    return "".join(f"{e}{n}" if n > 1 else e for e, n in zip(EL, v) if n)


def build_reach(maxops=MAXOPS, names=None, max_dec=MAX_DEC):
    """{delta vector: up to `max_dec` decompositions, shortest first}, computed once.

    One decomposition per delta is not enough: `+CH2O` puts a methoxy straight onto a ring carbon,
    while `+O` then `+CH2` gives a hydroxyl plus a methyl somewhere else. Same formula, different
    structures, and only one of them can be the truth. Combinations are generated shortest-first,
    so simply appending until `max_dec` keeps the cheapest and most chemically distinct routes.
    """
    names = list(PALETTE) if names is None else list(names)
    vecs = [PALETTE[n][0] for n in names]
    reach = {}
    for k in range(maxops + 1):
        for combo in itertools.combinations_with_replacement(range(len(vecs)), k):
            d = [0] * len(EL)
            for i in combo:
                vi = vecs[i]
                for j in range(len(EL)):
                    d[j] += vi[j]
            t = tuple(d)
            cur = reach.get(t)
            if cur is None:
                reach[t] = [tuple(names[i] for i in combo)]
            elif len(cur) < max_dec:
                cur.append(tuple(names[i] for i in combo))
    return {k: tuple(v) for k, v in reach.items()}


# ------------------------------------------------------------------ harvesting
def harvest_scaffolds(smiles, limit=25):
    """{canonical scaffold SMILES: mol} from `smiles`. Acyclic hits contribute the whole hit."""
    out = {}
    for smi in smiles[:limit]:
        m = Chem.MolFromSmiles(smi)
        if m is None or m.GetNumAtoms() == 0:
            continue
        try:
            sc = MurckoScaffold.GetScaffoldForMol(m)
        except Exception:
            sc = None
        if sc is None or sc.GetNumAtoms() == 0:
            sc = m                                  # 9.4% of truths are acyclic (GM.0a)
        try:
            key = Chem.MolToSmiles(sc)
        except Exception:
            continue
        if key not in out:
            out[key] = sc
    return out


def harvest_formulas(smiles, limit=25):
    """Distinct formula vectors of the hits, most frequent first. Already inside the mass window."""
    c = collections.Counter()
    for smi in smiles[:limit]:
        v = vec_of_smiles(smi)
        if v is not None:
            c[v] += 1
    return [v for v, _ in c.most_common()]


# ------------------------------------------------------------------ enumeration
def _apply(mols, name, beam=BEAM):
    """Apply one transform to every mol in `mols`; return the deduped, capped product frontier."""
    out = {}
    for rxn in _rxns(name):
        for m in mols:
            try:
                prods = rxn.RunReactants((m,))
            except Exception:
                continue
            for p in prods:
                q = p[0]
                try:
                    Chem.SanitizeMol(q)
                    s = Chem.MolToSmiles(q)
                except Exception:
                    continue
                if s not in out:
                    out[s] = q
    if len(out) <= beam:
        return list(out.values())
    return [out[k] for k in sorted(out)[:beam]]      # deterministic truncation


#   Transforms are applied socket-creators first. Murcko stripping removes every terminal O-H
#   (measured: 0 of 261 scaffolds carry one, against 65% of the truths), so each [OX2H]-gated
#   transform -- the sugars, +C2H2O, +SO3, +PO3, O-methylation -- can only fire once a +O has
#   made the hydroxyl. Applying the heavy groups first made all six of them dead on arrival.
_TIER = {"+O": 0, "+NH": 0}          # create the O-H / N-H sockets the rest attach to


def _order(decomposition):
    return sorted(decomposition, key=lambda n: (_TIER.get(n, 1), sum(abs(x) for x in PALETTE[n][0])))


def grow(scaffold, decomposition, target, beam=BEAM, cap=MAX_PRODUCTS):
    """Apply `decomposition` to `scaffold`; return canonical SMILES whose formula == `target`."""
    frontier = [scaffold]
    for name in _order(decomposition):
        frontier = _apply(frontier, name, beam)
        if not frontier:
            return []
    out = []
    for m in frontier:
        if elem_vec(m) == target:
            try:
                out.append(Chem.MolToSmiles(m))
            except Exception:
                pass
            if len(out) >= cap:
                break
    return out


def _union_scaffolds(sources, limits):
    """[(name, [smiles])] -> ([(key, mol, owner)] in round-robin order, per-source counts, owners).

    Two deliberate choices:

    * each source is harvested with **its own** limit, because `harvest_scaffolds` truncates at
      `limit` -- concatenating the lists first and passing one limit drops the later sources;
    * the walk order is round-robin, not concatenated, because `mutate_sources` returns early at
      `cap` or at the deadline.  Concatenated, the first source would claim the whole budget and
      the second would be dead on arrival.

    A scaffold found by several sources is owned by the first of them in `sources` order, so
    `owner == "eng"` means "absent from PubChem", independently of the round-robin interleaving.
    """
    harv = [(nm, harvest_scaffolds(smis, lim)) for (nm, smis), lim in zip(sources, limits)]
    counts = {nm: len(h) for nm, h in harv}
    owner = {}
    for nm, h in harv:
        for k in h:
            owner.setdefault(k, nm)
    lists = [list(h.items()) for _, h in harv]
    order, taken = [], set()
    for i in range(max((len(l) for l in lists), default=0)):
        for l in lists:
            if i < len(l) and l[i][0] not in taken:
                taken.add(l[i][0])
                order.append((l[i][0], l[i][1], owner[l[i][0]]))
    return order, counts, owner


def _union_formulas(sources, limits, max_formulas, contributes=None):
    """Formula targets over every source, most frequent first; each source keeps one of its own.

    A PubChem top-25 carries exactly one formula for 253 of the 400 test molecules (measured), and
    the retrieval sources all sit in the same +-10 ppm window, so this union is nearly always 1-3
    long and the guarantee below almost never fires.  It is here so a source with few hits cannot
    be voted out by a source with many.

    `contributes[i]` False means source `i` donates **scaffolds only**.  That is not a nicety: the
    analog source is drawn from a +-200 Da window, so its molecules are at the *wrong mass* and
    their formulas are not target hypotheses at all.  Letting them in would spend the whole budget
    growing scaffolds toward formulas the precursor mass rules out.
    """
    contributes = [True] * len(sources) if contributes is None else list(contributes)
    pairs = [(smis, lim) for (_, smis), lim, ok in zip(sources, limits, contributes) if ok]
    per = [harvest_formulas(smis, lim) for smis, lim in pairs]
    c = collections.Counter()
    for smis, lim in pairs:
        for smi in smis[:lim]:
            v = vec_of_smiles(smi)
            if v is not None:
                c[v] += 1
    keep = [v for v, _ in c.most_common()][:max_formulas]
    for p in per:
        if p and p[0] not in keep:
            if len(keep) < max_formulas:
                keep.append(p[0])
            else:
                keep = keep[:-1] + [p[0]]       # drop the least popular to make room
    return keep


def mutate_sources(sources, reach, per_source=(25, 40), max_formulas=6, cap=MAX_PRODUCTS,
                   beam=BEAM, deadline=None, stats=None, contributes_formulas=None,
                   max_steps=None):
    """[(name, [hit smiles])] -> ([candidate smiles], {candidate: owning source name}).

    `deadline` is an absolute `time.monotonic()` value; when it passes, whatever has been generated
    is returned rather than raised, so one pathological molecule cannot spend the run's budget.

    `contributes_formulas` is a per-source bool (default all True).  Pass False for a source whose
    molecules sit at a different mass than the target -- see `_union_formulas`.

    `max_steps` caps the decomposition length, either globally (an int) or per source
    ({source name: int}).  Measured on `pseudo_c3`: a delta realised in <= 2 transforms recovers
    the truth for 28.2% of the molecules whose scaffold is actually harvestable, against 3.7% at
    >= 3 steps and 0.0% at >= 5 -- and, just as importantly, a 1-step delta generates a median of
    7 candidates where a 2-step delta generates 95.  Since promotion needs the truth near the top
    of the mutator list rather than merely inside it, the shallow deltas are where this channel can
    actually be ranked, so capping depth buys accuracy and runtime at once.
    """
    sources = [(nm, list(smis)) for nm, smis in sources if smis]
    limits = list(per_source)[:len(sources)]
    while len(limits) < len(sources):
        limits.append(limits[-1] if limits else 25)
    if contributes_formulas is None:
        contributes_formulas = [True] * len(sources)
    contributes_formulas = list(contributes_formulas)[:len(sources)]
    while len(contributes_formulas) < len(sources):
        contributes_formulas.append(True)
    if max_steps is None or isinstance(max_steps, int):
        max_steps = {nm: max_steps for nm, _ in sources}
    scaffolds, sc_counts, _own = _union_scaffolds(sources, limits)
    formulas = _union_formulas(sources, limits, max_formulas, contributes_formulas)
    st = stats if stats is not None else collections.Counter()
    st["scaffolds"] += len(scaffolds)
    st["formulas"] += len(formulas)
    first = sources[0][0] if sources else ""
    for nm, n in sc_counts.items():
        st["sc_" + nm] += n
    for _k, _m, nm in scaffolds:
        if nm != first:
            st["sc_%s_only" % nm] += 1
    zero = tuple([0] * len(EL))
    seen, out, prov = set(), [], {}
    for _k, scaf, nm in scaffolds:
        sv = elem_vec(scaf)
        if sv is None:
            continue
        for tv in formulas:
            if deadline is not None and time.monotonic() > deadline:
                st["hit_deadline"] += 1
                return out, prov
            delta = tuple(t - s for t, s in zip(tv, sv))
            if delta == zero:
                st["pairs_identity"] += 1
                continue
            decs = reach.get(delta)
            if decs is None:
                st["pairs_unreachable"] += 1
                continue
            lim = max_steps.get(nm)
            if lim is not None:
                decs = tuple(d for d in decs if len(d) <= lim)
                if not decs:
                    st["pairs_too_deep"] += 1
                    continue
            st["pairs_reachable"] += 1
            for dec in decs:
                for s in grow(scaf, dec, tv, beam, cap - len(out)):
                    if s not in seen:
                        seen.add(s)
                        out.append(s)
                        prov[s] = nm
                if len(out) >= cap:
                    st["hit_cap"] += 1
                    return out, prov
    return out, prov


def mutate(hit_smiles, reach, max_scaffolds=25, max_formulas=6, cap=MAX_PRODUCTS,
           beam=BEAM, stats=None):
    """Single-source wrapper.  Output order is identical to `mutate_sources` with one source."""
    out, _ = mutate_sources([("pc", hit_smiles)], reach, per_source=(max_scaffolds,),
                            max_formulas=max_formulas, cap=cap, beam=beam, stats=stats)
    return out


if __name__ == "__main__":
    import time

    t0 = time.time()
    R = build_reach()
    print(f"reachable deltas (depth<={MAXOPS}, {len(PALETTE)} transforms): {len(R):,} "
          f"in {time.time()-t0:.1f}s")
    demo = ["COc1cc(C2Oc3cc(O)ccc3C2)ccc1O", "Oc1ccc2c(c1)oc(-c1ccc(O)cc1)c(O)c2=O",
            "CC(=O)Oc1ccccc1C(=O)O"]
    st = collections.Counter()
    t0 = time.time()
    got = mutate(demo, R, stats=st)
    print(f"demo: {len(got)} candidates in {time.time()-t0:.1f}s; stats {dict(st)}")
    for s in got[:5]:
        print("   ", s)
