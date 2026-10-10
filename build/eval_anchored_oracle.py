"""Local oracle for mutator v2 (c3v2 plan I0/I0b/I1/I2): anchored round trips on pseudo_c3.

Production anchors are spectral analogs from engine A, which cannot be computed offline (no
train.parquet, no pool SMILES here).  This harness builds the anchors synthetically instead: for each
`pseudo_c3` truth T it applies 1-2 random signed moves to T to make a "database relative" A, then
asks the v2 mutator to get back from A to T using only the exact mass delta, as production would.

That measures the generator, not the gate:
  * completeness -- is T among the products of A at all (forward AND reverse moves)?
  * pruning loss -- with a score-guided frontier, how many recoverable T are lost at each tau?
  * rank        -- where T lands among A's products under a noisy oracle fingerprint score.

The oracle score is f.z with z_i = +1 for T's Morgan-r2 bits and -1 otherwise, plus Gaussian noise
sigma (sigma=0 is a perfect fingerprint predictor; FPNet is far from that, so read sigma>0).  tau is
in the same units (one bit ~ 1), so tau=4 tolerates an intermediate losing ~4 net bits.

Synthetic relatives are cleaner than real ones (one exact edit, no other differences): treat the
numbers as a ceiling for the generator and a fair test of the pruning rule, not as a recovery rate.

usage: python build/eval_anchored_oracle.py [--limit N] [--depth 1|2] [--sigma S] [--taus 0,2,4,8]
                                            [--direction any|up]
  --direction up: the relative is the truth with groups REMOVED, so the mutator has to add them back
  at the right site -- the realistic hard case (a known parent, a novel decorated truth).
"""
from __future__ import annotations

import collections
import os
import random
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import inchi, rdFingerprintGenerator

import delta_m_mutator as M

RDLogger.DisableLog("rdApp.*")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MG = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)


def ik14(smi):
    m = Chem.MolFromSmiles(smi)
    return None if m is None else inchi.MolToInchiKey(m)[:14]


def fp(smi):
    m = Chem.MolFromSmiles(smi)
    return None if m is None else _MG.GetFingerprintAsNumPy(m).astype(np.float64)


def oracle_scorer(truth_smi, sigma, rng):
    t = fp(truth_smi)
    z = 2.0 * t - 1.0 + (rng.normal(0.0, sigma, t.shape) if sigma > 0 else 0.0)

    def score(smiles):
        out = np.full(len(smiles), -1e9)
        for i, s in enumerate(smiles):
            f = fp(s)
            if f is not None:
                out[i] = float(f @ z)
        return out
    return score


def make_relative(truth, depth, rng, names):
    """Apply `depth` random signed moves to the truth -> (relative smiles, moves) or None."""
    m = Chem.MolFromSmiles(truth)
    moves = []
    for _ in range(depth):
        opts = []
        for n in rng.sample(names, len(names)):
            prods = []
            for q in M._run_move(m, n):
                try:
                    Chem.SanitizeMol(q)
                    s = Chem.MolToSmiles(q)
                except Exception:
                    continue
                if s and "." not in s:
                    prods.append(s)
            if prods:
                opts.append((n, rng.choice(sorted(set(prods)))))
            if len(opts) >= 3:
                break
        if not opts:
            return None
        n, s = rng.choice(opts)
        moves.append(n)
        m = Chem.MolFromSmiles(s)
    rel = Chem.MolToSmiles(m)
    return (rel, moves) if ik14(rel) != ik14(truth) else None


def main(argv):
    limit, depth, sigma, taus, direction = 150, 1, 1.0, [0.0, 2.0, 4.0, 8.0], "any"
    i = 0
    while i < len(argv):
        k = argv[i]
        if k == "--limit":
            limit = int(argv[i + 1])
        elif k == "--depth":
            depth = int(argv[i + 1])
        elif k == "--sigma":
            sigma = float(argv[i + 1])
        elif k == "--direction":
            direction = argv[i + 1]
        elif k == "--taus":
            taus = [float(x) for x in argv[i + 1].split(",")]
        i += 2
    rng, nrng = random.Random(0), np.random.default_rng(0)
    df = pd.read_parquet(os.path.join(ROOT, "p1_novel", "pseudo_c3.parquet"))
    df = df[df.in_scope].head(limit) if "in_scope" in df else df.head(limit)
    reach = M.build_reach_signed(max(2, depth))
    idx = M.mass_index(reach)
    names = M.signed_names()
    if direction == "up":       # relatives lose heavy atoms -> the way back adds them
        names = [n for n in names if sum(M.signed_vec(n)) - M.signed_vec(n)[1] < 0]
    print("pseudo_c3 round trips: %d truths, relative depth %d, sigma %.1f, taus %s, direction %s"
          % (len(df), depth, sigma, taus, direction))
    print("signed reach: %d deltas over %d moves" % (len(reach), len(names)))

    rows = []
    for r in df.itertuples():
        rel = make_relative(r.can, depth, rng, names)
        if rel is None:
            rows.append(dict(mid=r.molecule_id, made=False))
            continue
        a, moves = rel
        dm = M.vec_mass(M.vec_of_smiles(r.can)) - M.vec_mass(M.vec_of_smiles(a))
        deltas = M.match_mass_delta(dm, idx, 0.003)
        tk = ik14(r.can)
        row = dict(mid=r.molecule_id, made=True, moves="+".join(moves),
                   reverse=any(n not in M.ADDITIVE_V2 for n in moves), n_deltas=len(deltas),
                   delta_hit=M.vec_of_smiles(r.can) is not None and
                   tuple(x - y for x, y in zip(M.vec_of_smiles(r.can), M.vec_of_smiles(a))) in deltas)
        score = oracle_scorer(r.can, sigma, nrng)
        for tau in [None] + taus:
            st = collections.Counter()
            t0 = time.time()
            out = M.mutate_anchored([(a, deltas)], reach, beam=200, cap=3000, max_steps=depth,
                                    scorer=None if tau is None else score, tau=tau, stats=st)
            secs = time.time() - t0
            smis = [s for s, _, _ in out]
            keys = [ik14(s) for s in smis]
            lab = "full" if tau is None else "t%g" % tau
            row[lab + "_hit"] = tk in keys
            row[lab + "_n"] = len(smis)
            row[lab + "_s"] = secs
            if tk in keys:
                sc = score(smis)
                ts = sc[keys.index(tk)]
                row[lab + "_rank"] = int(1 + np.sum(sc > ts))
                row[lab + "_beats_anchor"] = bool(ts > score([a])[0])
        rows.append(row)
    d = pd.DataFrame(rows)
    made = d[d.made]
    print("\nrelatives made: %d/%d (%.0f%% use a reverse move); exact delta among mass matches: %.1f%%"
          % (len(made), len(d), 100 * made.reverse.mean(), 100 * made.delta_hit.mean()))
    print("\n%-6s%9s%10s%10s%12s%13s%9s" % ("mode", "recall", "lost", "rank<=3", "cands med",
                                           "fz(T)>fz(A)", "s/mol"))
    base = made["full_hit"]
    for lab in ["full"] + ["t%g" % t for t in taus]:
        h = made[lab + "_hit"]
        lost = int((base & ~h).sum())
        rk = made.loc[h, lab + "_rank"] if lab + "_rank" in made else pd.Series(dtype=float)
        ba = made.loc[h, lab + "_beats_anchor"].mean() if h.any() else float("nan")
        print("%-6s%8.1f%%%10d%9.1f%%%12.0f%12.1f%%%9.2f"
              % (lab, 100 * h.mean(), lost, 100 * (rk <= 3).sum() / len(made),
                 made[lab + "_n"].median(), 100 * ba, made[lab + "_s"].mean()))
    os.makedirs(os.path.join(ROOT, "scratch"), exist_ok=True)
    out = os.path.join(ROOT, "scratch", "c3v2_anchored_oracle_d%d_s%g_%s.csv" % (depth, sigma, direction))
    d.to_csv(out, index=False)
    print("\nper-molecule rows ->", out)
    return d


if __name__ == "__main__":
    main(sys.argv[1:])
