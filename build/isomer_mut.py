"""Mass-preserving isomer operators (c3-iso I1): one parent SMILES -> children with the IDENTICAL molecular formula.

Pure RDKit. Stereo is stripped first (the metric ignores it). Every child is sanitized and its formula compared to the
parent's (`CalcMolFormula`); a mismatch is dropped and counted in `stats['formula_drop']` (it is never returned).

Operators (<= 2 moves, labelled per child):
  subst_shift   a terminal/acyclic group (OH, OMe, Me, X, NH2, COOH, O-glycoside, ring B of a flavonoid ...) detached from
                a ring atom and re-attached to another ring atom of the SAME ring system that carries an H
  methyl_swap   Ar-O-CH3 <-> Ar-CH2-OH (also N: Ar-NH-CH3 <-> Ar-CH2-NH2) on the same attachment atom
  group_move    O/N-linked CH3, acyl C(=O)R or sp3 C (sugar) moved from its O/N to another free OH/NH of the same side
                (methyl migration, glycoside / ester position)
  db_shift      a non-aromatic C=C moved one position (C1=C2-C3H -> C1H-C2=C3)
  chain_iso     a terminal CH3 on an sp3 carbon moved to an sp3 carbon with an H within 2 bonds (n-alkyl <-> iso-alkyl)
Biogenetic prior (ordering only): the attachment atom is an O/N (2), or ortho to an O/N-bearing atom (1), else 0.
"""
from __future__ import annotations

import time

from rdkit import Chem, RDLogger
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

RDLogger.DisableLog("rdApp.*")

MAX_FRAG = 25  # heavy atoms of a moved group


def _parse(smi):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if m is None:
        return None
    Chem.RemoveStereochemistry(m)
    return m


def canon(smi):
    m = _parse(smi)
    return Chem.MolToSmiles(m) if m is not None else None


def formula(m):
    return CalcMolFormula(m)


def ik14_plain(smi):
    """RDKit InChIKey[:14] without tautomer canonicalisation (cheap, for dedupe)."""
    m = _parse(smi)
    if m is None:
        return None
    try:
        return Chem.MolToInchiKey(m)[:14]
    except Exception:
        return None


def _side(m, a, b):
    """Atoms on b's side of the bridge bond a-b (BFS not crossing the bond)."""
    seen, stack = {b}, [b]
    while stack:
        x = stack.pop()
        for n in m.GetAtomWithIdx(x).GetNeighbors():
            j = n.GetIdx()
            if j in seen or (x == b and j == a):
                continue
            seen.add(j)
            stack.append(j)
    return seen


def _ring_system(m, a):
    ri = m.GetRingInfo()
    seen, stack = {a}, [a]
    while stack:
        x = stack.pop()
        for bd in m.GetAtomWithIdx(x).GetBonds():
            if bd.IsInRing():
                j = bd.GetOtherAtomIdx(x)
                if j not in seen:
                    seen.add(j)
                    stack.append(j)
    return seen


def _hetero(at):
    return at.GetAtomicNum() in (7, 8)


def _prior(m, t):
    at = m.GetAtomWithIdx(t)
    if _hetero(at):
        return 2
    return 1 if any(_hetero(n) for n in at.GetNeighbors()) else 0


def _reattach(m, a, b, t):
    """Detach b from a, attach b to t. Returns sanitized canonical SMILES or None."""
    rw = Chem.RWMol(m)
    rw.RemoveBond(a, b)
    rw.AddBond(b, t, Chem.BondType.SINGLE)
    ta, tt = rw.GetAtomWithIdx(a), rw.GetAtomWithIdx(t)
    if ta.GetNoImplicit() or ta.GetNumExplicitHs():
        ta.SetNumExplicitHs(ta.GetNumExplicitHs() + 1)
    if tt.GetNoImplicit() or tt.GetNumExplicitHs():
        tt.SetNumExplicitHs(max(0, tt.GetNumExplicitHs() - 1))
    return _finish(rw)


def _finish(rw):
    try:
        mol = rw.GetMol()
        Chem.SanitizeMol(mol)
        smi = Chem.MolToSmiles(mol)
        return smi
    except Exception:
        return None


def _totalH(at):
    return at.GetTotalNumHs()


def _one_moves(m):
    """All single moves of mol m -> list of (op, child_smiles, prior)."""
    out = []
    # --- subst_shift: bridge bond a-b, a in ring, same ring system target with H
    seen_rs = {}
    for bd in m.GetBonds():
        if bd.IsInRing() or bd.GetBondType() != Chem.BondType.SINGLE:
            continue
        for a, b in ((bd.GetBeginAtomIdx(), bd.GetEndAtomIdx()), (bd.GetEndAtomIdx(), bd.GetBeginAtomIdx())):
            aa = m.GetAtomWithIdx(a)
            if not aa.IsInRing():
                continue
            S = _side(m, a, b)
            if len(S) > MAX_FRAG or a in S:
                continue
            rs = seen_rs.setdefault(a, _ring_system(m, a))
            for t in rs:
                if t == a:
                    continue
                at = m.GetAtomWithIdx(t)
                if _totalH(at) < 1 or at.GetAtomicNum() not in (6, 7):
                    continue
                s = _reattach(m, a, b, t)
                if s:
                    out.append(("subst_shift", s, _prior(m, t)))
    # --- methyl_swap and group_move
    for bd in m.GetBonds():
        if bd.IsInRing() or bd.GetBondType() != Chem.BondType.SINGLE:
            continue
        for a, b in ((bd.GetBeginAtomIdx(), bd.GetEndAtomIdx()), (bd.GetEndAtomIdx(), bd.GetBeginAtomIdx())):
            aa, bb = m.GetAtomWithIdx(a), m.GetAtomWithIdx(b)
            # methyl_swap, forward: a(C)-X-CH3, X in O,N  ->  a-CH2-XH
            if aa.GetAtomicNum() == 6 and bb.GetAtomicNum() in (7, 8) and bb.GetDegree() == 2:
                me = [n.GetIdx() for n in bb.GetNeighbors() if n.GetIdx() != a][0]
                mea = m.GetAtomWithIdx(me)
                if mea.GetAtomicNum() == 6 and mea.GetDegree() == 1 and not mea.GetIsAromatic():
                    rw = Chem.RWMol(m)
                    rw.RemoveBond(a, b)
                    rw.AddBond(a, me, Chem.BondType.SINGLE)
                    s = _finish(rw)
                    if s:
                        out.append(("methyl_swap", s, _prior(m, a)))
            # methyl_swap, reverse: a(C)-CH2-XH (CH2 has degree 2, XH degree 1)  ->  a-X-CH3
            if (aa.GetAtomicNum() == 6 and bb.GetAtomicNum() == 6 and not bb.GetIsAromatic() and bb.GetDegree() == 2
                    and bb.GetTotalNumHs() == 2):
                x = [n for n in bb.GetNeighbors() if n.GetIdx() != a][0]
                if x.GetAtomicNum() in (7, 8) and x.GetDegree() == 1:
                    rw = Chem.RWMol(m)
                    rw.RemoveBond(a, b)
                    rw.AddBond(a, x.GetIdx(), Chem.BondType.SINGLE)
                    s = _finish(rw)
                    if s:
                        out.append(("methyl_swap", s, _prior(m, a)))
            # group_move: a = O/N linker (degree 2), b = CH3 / sp3 C (sugar) / carbonyl C; move b to another O/N-H
            if aa.GetAtomicNum() in (7, 8) and aa.GetDegree() == 2 and bb.GetAtomicNum() == 6 \
                    and not bb.GetIsAromatic():
                sp3 = bb.GetHybridization() == Chem.HybridizationType.SP3
                carbonyl = any(x.GetAtomicNum() == 8 and m.GetBondBetweenAtoms(b, x.GetIdx()).GetBondType()
                               == Chem.BondType.DOUBLE for x in bb.GetNeighbors())
                if not (sp3 or carbonyl):
                    continue
                S = _side(m, a, b)
                if len(S) > MAX_FRAG:
                    continue
                for t in range(m.GetNumAtoms()):
                    if t in S or t == a:
                        continue
                    at = m.GetAtomWithIdx(t)
                    if at.GetAtomicNum() not in (7, 8) or _totalH(at) < 1 or at.GetFormalCharge() != 0:
                        continue
                    s = _reattach(m, a, b, t)
                    if s:
                        out.append(("group_move", s, 2))
    # --- db_shift: C1=C2-C3(H)  ->  C1(H)-C2=C3
    for bd in m.GetBonds():
        if bd.GetBondType() != Chem.BondType.DOUBLE or bd.GetIsAromatic():
            continue
        if bd.GetBeginAtom().GetAtomicNum() != 6 or bd.GetEndAtom().GetAtomicNum() != 6:
            continue
        for c1, c2 in ((bd.GetBeginAtomIdx(), bd.GetEndAtomIdx()), (bd.GetEndAtomIdx(), bd.GetBeginAtomIdx())):
            for n in m.GetAtomWithIdx(c2).GetNeighbors():
                c3 = n.GetIdx()
                if c3 == c1 or n.GetAtomicNum() != 6 or n.GetIsAromatic() or n.GetTotalNumHs() < 1:
                    continue
                if m.GetBondBetweenAtoms(c2, c3).GetBondType() != Chem.BondType.SINGLE:
                    continue
                rw = Chem.RWMol(m)
                rw.GetBondBetweenAtoms(c1, c2).SetBondType(Chem.BondType.SINGLE)
                rw.GetBondBetweenAtoms(c2, c3).SetBondType(Chem.BondType.DOUBLE)
                for i in (c1, c3):
                    at = rw.GetAtomWithIdx(i)
                    if at.GetNoImplicit() or at.GetNumExplicitHs():
                        at.SetNumExplicitHs(max(0, at.GetNumExplicitHs() + (1 if i == c1 else -1)))
                s = _finish(rw)
                if s:
                    out.append(("db_shift", s, 0))
    # --- chain_iso: terminal CH3 on sp3 C a -> sp3 C within 2 bonds with an H
    for at in m.GetAtoms():
        if at.GetAtomicNum() != 6 or at.GetDegree() != 1 or at.GetIsAromatic() or at.GetTotalNumHs() != 3:
            continue
        b = at.GetIdx()
        a = at.GetNeighbors()[0]
        if a.GetAtomicNum() != 6 or a.GetIsAromatic() or a.GetHybridization() != Chem.HybridizationType.SP3:
            continue
        a = a.GetIdx()
        near = set()
        for n1 in m.GetAtomWithIdx(a).GetNeighbors():
            if n1.GetIdx() != b:
                near.add(n1.GetIdx())
                for n2 in n1.GetNeighbors():
                    near.add(n2.GetIdx())
        near -= {a, b}
        for t in near:
            ta = m.GetAtomWithIdx(t)
            if (ta.GetAtomicNum() == 6 and not ta.GetIsAromatic() and ta.GetHybridization() == Chem.HybridizationType.SP3
                    and ta.GetTotalNumHs() >= 1):
                s = _reattach(m, a, b, t)
                if s:
                    out.append(("chain_iso", s, 0))
    return out


def _moves_checked(smi, f0, stats):
    """Single moves of SMILES `smi`, formula-checked against f0 (parent's)."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return []
    res = []
    for op, s, pr in _one_moves(m):
        c = Chem.MolFromSmiles(s)
        if c is None:
            stats["bad_smiles"] = stats.get("bad_smiles", 0) + 1
            continue
        if formula(c) != f0:
            stats["formula_drop"] = stats.get("formula_drop", 0) + 1
            continue
        res.append((op, Chem.MolToSmiles(c), pr))
    return res


def mutate(parent_smi, max_moves=2, cap=300, deadline=None, stats=None, l2_from=40):
    """One parent -> [(child_smiles, op_label, n_moves)] (unique, != parent), L1 first (prior-ordered), then L2.

    L2 expands the `l2_from` best-prior L1 children. `deadline` is a time.monotonic() value."""
    st = stats if stats is not None else {}
    pm = _parse(parent_smi)
    if pm is None:
        return []
    f0 = formula(pm)
    p0 = Chem.MolToSmiles(pm)
    seen = {p0}
    out = []
    l1 = _moves_checked(p0, f0, st)
    l1.sort(key=lambda x: -x[2])
    l1u = []
    for op, s, pr in l1:
        if s not in seen:
            seen.add(s)
            l1u.append((op, s, pr))
            out.append((s, op, 1))
    if max_moves >= 2:
        for op1, s1, _ in l1u[:l2_from]:
            if (cap and len(out) >= cap) or (deadline and time.monotonic() > deadline):
                break
            l2 = _moves_checked(s1, f0, st)
            l2.sort(key=lambda x: -x[2])
            for op2, s2, pr in l2:
                if s2 not in seen:
                    seen.add(s2)
                    out.append((s2, f"{op1}+{op2}", 2))
    return out[:cap] if cap else out


def mutate_many(parents, cap=300, deadline=None, exclude_keys=(), stats=None, max_moves=2):
    """Fan out over several parents (fused rank order) with ONE cap per molecule.

    Round-robin across parents so a long parent does not eat the cap. Dedupe by plain IK14, drop children whose IK14
    is in `exclude_keys` (the retrieved candidates). Returns [(child_smiles, ik14, op_label, parent_smiles, n_moves)]."""
    st = stats if stats is not None else {}
    per = []
    for p in parents:
        if deadline and time.monotonic() > deadline:
            break
        per.append((p, mutate(p, max_moves=max_moves, cap=0, deadline=deadline, stats=st)))
    keys, out = set(exclude_keys), []
    keys.update(k for k in (ik14_plain(p) for p in parents) if k)  # a child tautomer-equal to a parent is no new candidate
    # L1 children of all parents first (round robin), then L2
    for nm in (1, 2):
        lists = [(p, [c for c in ch if c[2] == nm]) for p, ch in per]
        i = 0
        while any(len(l) > i for _, l in lists):
            for p, l in lists:
                if i < len(l):
                    s, op, n = l[i]
                    k = ik14_plain(s)
                    if k and k not in keys:
                        keys.add(k)
                        out.append((s, k, op, p, n))
                        if len(out) >= cap:
                            return out
            i += 1
    return out
