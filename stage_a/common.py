"""Stage A shared helpers (iso-rank). Pure functions; no truth is read here except in split_of / truth key helpers,
which take a truth SMILES passed in by the analysis step."""
from __future__ import annotations

import hashlib
import re
import time

import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))

N_PARENTS, CAP, DEADLINE, TOP = 10, 600, 120.0, 150
V22 = dict(METRIC_FUSION=True, ICE_FALLBACK=True, FUSION_WINDOW=40, REL_PROMOTE=150.0, FRAG_GAP=True, E2R2=True)


def scaffold_smiles(smiles):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return ''
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m)
    except Exception:
        return ''


def id_parity(mid):
    """Parity of the last digit run of a molecule_id (C3_00007 -> 1, VAL00012 -> 0); md5 parity if no digits."""
    d = re.findall(r"\d+", str(mid))
    if d:
        return int(d[-1]) % 2
    return int(hashlib.md5(str(mid).encode()).hexdigest(), 16) % 2


def split_of(truth_smiles, mid):
    """Pre-registered split: int(md5(Murcko scaffold SMILES), 16) % 2 (0 -> 'A'); acyclic truths (empty scaffold) are
    split by molecule_id parity instead. The truth only groups molecules; it never selects children."""
    sc = scaffold_smiles(truth_smiles)
    p = id_parity(mid) if sc == '' else int(hashlib.md5(sc.encode()).hexdigest(), 16) % 2
    return 'A' if p == 0 else 'B'


def gen_edges(parents, cap=CAP, deadline_s=DEADLINE, exclude_keys=(), stats=None, max_moves=2):
    """isomer_mut.mutate_many's selection (same order, cap, dedupe) PLUS the full edge table for multiplicity.

    -> (selected, pairs)  selected = [(smiles, ik14, op, parent_smiles, n_moves)] exactly as mutate_many returns;
    pairs = {child ik14: sorted [(parent_index, op)]} over EVERY edge generated (selected children only), so a child
    reached by two parents/operators has multiplicity 2 even though mutate_many keeps one copy."""
    import isomer_mut as im
    st = stats if stats is not None else {}
    t0 = time.monotonic()
    deadline = t0 + deadline_s
    per = []
    for p in parents:
        if time.monotonic() > deadline:
            st["deadline_hit"] = 1
            break
        per.append((p, im.mutate(p, max_moves=max_moves, cap=0, deadline=deadline, stats=st)))
    keys, out = set(exclude_keys), []
    keys.update(k for k in (im.ik14_plain(p) for p in parents) if k)
    kcache = {}

    def key(s):
        if s not in kcache:
            kcache[s] = im.ik14_plain(s)
        return kcache[s]

    done = False
    for nm in (1, 2):
        lists = [(p, [c for c in ch if c[2] == nm]) for p, ch in per]
        i = 0
        while not done and any(len(l) > i for _, l in lists):
            for p, l in lists:
                if time.monotonic() > deadline + im.SELECT_GRACE:
                    st["deadline_hit"] = 1
                    done = True
                    break
                if i < len(l):
                    s, op, n = l[i]
                    k = key(s)
                    if k and k not in keys:
                        keys.add(k)
                        out.append((s, k, op, p, n))
                        if len(out) >= cap:
                            done = True
                            break
            i += 1
        if done:
            break
    sel = {c[1] for c in out}
    pairs = {}
    for pi, (p, ch) in enumerate(per):
        for s, op, n in ch:
            k = key(s)
            if k in sel:
                pairs.setdefault(k, set()).add((pi, op))
    return out, {k: sorted(v) for k, v in pairs.items()}
