"""Track 4 probe generator: patch the 0.405 supernext notebook, one flag per LB probe.

Never hand-edit the generated .ipynb -- change this script and regenerate. The source notebook
is read, never written. Outputs are cleared so every variant uploads cleanly.

Always applied (no score effect, instrumentation only):
  * POOL_POP is loaded once before the BASE loop, and a pool key -> popularity map (POPK) is
    recorded over every in-window pool candidate V.run returns
  * the ranker's own top-1 per molecule (RAW_TOP1) is recorded before the popularity re-sort
  * counters: `pop stats` (min/median/p99 of the applied prior term), candidates promoted into
    the top TOPN, POPK size, and in the fusion cell `pop-fused top-1 moved` / `c1 frozen`

Flags (Track 4 roadmap, Wave A):
  --pop-precut       A1  apply the prior to the top POP_UNION (200) ranker candidates, THEN cut
                         to TOPN (the base applies it after the cut, so it can only re-order)
  --pop-fused [MU]   A2  after fuse2, add MU*pop (default 0.15) to z(RRF) before the ICE/GL rerank
  --c1-freeze        A4  molecules with lib_max >= LIB_TAU keep the ranker's top-1 at rank 1
  --mu MU            A3  set POOLPOP_MU (base 0.15)
  --gen-only         A5  POOLPOP_MU = 0 but keep the '|gen' formula isolation (attribution only)

Track 4b flags (specs/track4_priors_probes/np_signals_plan.md, gate_report.md amendment A-NB.1):
  --nplike MU        T4B-N  after the PubChem list's ICE/GL re-order, re-sort its candidates by
                            z(-rank) + MU*z(Ertl NP-likeness)
  --classcons MU     T4B-C  same, with MU*z(superclass agreement with BASE's top-5 pool candidates);
                            labels from the NP knowledge table (scorer_key) or the nearest LOTUS
                            labelled NP-table structure (Tanimoto >= 0.7); needs only the NP knowledge table; unlabelled = 0
  pc_fz[0] (the pure spectral top score behind the rel merge gate) is never changed.

usage: python build/patch_supernext.py [flags] [--src in.ipynb] [--out out.ipynb]
"""
from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "evenda-casmi-inferencesupernext.ipynb")


def src_of(cell):
    return "".join(cell["source"])


def set_src(cell, text):
    cell["source"] = text.splitlines(keepends=True)


def must_replace(text, old, new, what):
    if old not in text:
        raise SystemExit(f"PATCH FAILED: anchor not found for {what}:\n  {old[:120]}")
    if text.count(old) != 1:
        raise SystemExit(f"PATCH FAILED: anchor for {what} occurs {text.count(old)} times")
    return text.replace(old, new)


def find(nb, pred, what):
    hits = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code" and pred(src_of(c))]
    if len(hits) != 1:
        raise SystemExit(f"PATCH FAILED: {what} matched {len(hits)} cells")
    return hits[0]


# ---------------------------------------------------------------- config cell
def patch_config(t, a):
    if a.mu is not None or a.gen_only:
        mu = 0.0 if a.gen_only else a.mu
        t = must_replace(t, "POP_LAM, POP_UNION, POOLPOP_MU = 0.25, 200, 0.15",
                         f"POP_LAM, POP_UNION, POOLPOP_MU = 0.25, 200, {mu!r}", "POOLPOP_MU")
    return t


# ---------------------------------------------------------------- BASE cell (C7)
PRE_LOOP = """BASE, n_err = {}, 0
# === TRACK 4 (patch_supernext.py) ===
POPK, RAW_TOP1, POP_LOG, N_PROMOTED = {}, {}, [], 0
POOL_POP = None
if POP_ROOT:
    try:
        POOL_POP = np.load(os.path.join(POP_ROOT, 'pool_lsid.npy')).astype(np.float64) + np.load(os.path.join(POP_ROOT, 'pool_lpmid.npy')).astype(np.float64)
        print('POOL_POP', len(POOL_POP), 'entries; min/median/max', float(POOL_POP.min()), float(np.median(POOL_POP)), float(POOL_POP.max()))
    except Exception as _e:
        POOL_POP = None; print('POOL_POP load failed', repr(_e))
TOPN_PRE = @TOPN_PRE@
"""

RECORD_POPK = """                score = rank_score(np.concatenate([X, F], 1), list(FEATURES) + list(names))
                if POOL_POP is not None:
                    try:
                        for k_, p_ in zip(C.key, C.pid):
                            if p_ >= 0 and k_: POPK[k_] = float(POOL_POP[p_])
                    except Exception as _e:
                        print('POPK record failed', mid, repr(_e))
"""

POP_BLOCK_HEAD = """        RAW_TOP1[mid] = smis[0] if smis else None
        _raw_top = set(keys[:TOPN])
        if POP_ROOT and POOLPOP_MU @CMP@ 0 and len(scs) > 1:"""

POP_LOG_LINE = """                f_ = z_ + POOLPOP_MU * np.where(isg_, 0.0, POOL_POP[np.maximum(pid_, 0)])
                POP_LOG.extend((f_ - z_)[~isg_][:TOPN].tolist())
"""

CUT = """    smis, keys, scs, forms = smis[:TOPN], keys[:TOPN], scs[:TOPN], forms[:TOPN]
    N_PROMOTED += len(set(keys) - _raw_top)
    BASE[mid] = (smis, keys, lib_max, scs, forms)
"""

BASE_TAIL = """print('errors', n_err)
_pl = np.asarray(POP_LOG) if POP_LOG else np.zeros(1)
print(f'pop stats: n={len(POP_LOG)} min={_pl.min():.3f} median={np.median(_pl):.3f} p99={np.percentile(_pl, 99):.3f} '
      f'| promoted into top{TOPN}: {N_PROMOTED} | TOPN_PRE {TOPN_PRE} | POPK {len(POPK)} keys')
"""


def patch_base(t, a):
    topn_pre = "max(TOPN, int(POP_UNION)) if (POP_ROOT and POOLPOP_MU > 0) else TOPN" if a.pop_precut else "TOPN"
    t = must_replace(t, "BASE, n_err = {}, 0\n", PRE_LOOP.replace("@TOPN_PRE@", topn_pre), "pre-loop init")
    t = must_replace(t, "    smis, keys, lib_max, scs, forms, pids_ = [], [], 0.0, [], [], []\n",
                     "    smis, keys, lib_max, scs, forms, pids_ = [], [], 0.0, [], [], []\n    _raw_top = set()\n",
                     "per-molecule init")
    t = must_replace(t, "                score = rank_score(np.concatenate([X, F], 1), list(FEATURES) + list(names))\n",
                     RECORD_POPK, "POPK record")
    t = must_replace(t, "                    if len(smis) >= TOPN: break",
                     "                    if len(smis) >= TOPN_PRE: break", "TOPN_PRE cut")
    t = must_replace(t, "        if POP_ROOT and POOLPOP_MU > 0 and len(scs) > 1:",
                     POP_BLOCK_HEAD.replace("@CMP@", ">=" if a.gen_only else ">"), "pop block head")
    t = must_replace(t, "                if 'POOL_POP' not in locals():", "                if POOL_POP is None:", "POOL_POP guard")
    t = must_replace(t, "                f_ = z_ + POOLPOP_MU * np.where(isg_, 0.0, POOL_POP[np.maximum(pid_, 0)])\n",
                     POP_LOG_LINE, "pop log")
    t = must_replace(t, "    BASE[mid] = (smis, keys, lib_max, scs, forms)\n", CUT, "TOPN cut")
    t = must_replace(t, "print('errors', n_err)", BASE_TAIL.rstrip("\n"), "pop stats")   # last line of the cell: no newline
    return t


# ---------------------------------------------------------------- ICE/GL cell: Track 4b NP signals
NP_BLOCK = r"""# === TRACK 4b (patch_supernext.py): NP signals re-order the PubChem list inside its candidates ===
NP_MU_LIKE, NP_MU_CLS = @MU_LIKE@, @MU_CLS@
NP_STATS = dict(molecules=0, reordered=0, top1_changed=0, labelled_cands=0, cands=0, base_labelled=0)
_NTERMS = []
if (NP_MU_LIKE > 0 or NP_MU_CLS > 0) and isinstance(PC, dict):
    try:
        from rdkit import Chem as _NC, DataStructs as _NDS, RDLogger as _NRL
        from rdkit.Chem import RDConfig as _NRC, rdFingerprintGenerator as _NFG
        _NRL.DisableLog('rdApp.*')
        _NGEN = _NFG.GetMorganGenerator(radius=2, fpSize=2048)
        def _nfp(s):
            m = _NC.MolFromSmiles(s)
            return None if m is None else _NGEN.GetFingerprint(m)
        def _nz(x):
            x = np.asarray(x, np.float64); ok = np.isfinite(x); out = np.zeros(len(x))
            if ok.sum() >= 2 and x[ok].std() > 0:
                out[ok] = (x[ok] - x[ok].mean()) / x[ok].std()
            return out
        if NP_MU_LIKE > 0:
            sys.path.append(os.path.join(_NRC.RDContribDir, 'NP_Score'))
            import npscorer as _NPS
            _NPM = _NPS.readNPModel()
            print('TRACK4b nplike: NP_Score model loaded from', _NRC.RDContribDir)
        if NP_MU_CLS > 0:
            _npt = pd.read_parquet(find('natural_products.parquet'), columns=['scorer_key', 'smiles', 'np_superclass'])
            _npt = _npt[_npt.np_superclass.notna()]
            _NLAB = dict(zip(_npt.scorer_key, _npt.np_superclass))
            _NBANK, _NLL = [], []   # kNN bank = the labelled NP-table structures (as in the Colab benchmark)
            for _s, _l in zip(_npt.smiles, _npt.np_superclass):
                _fp = _nfp(_s)
                if _fp is not None:
                    _NBANK.append(_fp); _NLL.append(_l)
            print('TRACK4b classcons: NP-table labels', len(_NLAB), '| kNN bank', len(_NBANK))
            _NCACHE = {}
            def _nlab(s, k):
                if k in _NLAB: return _NLAB[k]
                if k not in _NCACHE:
                    fp = _nfp(s); lab = None
                    if fp is not None:
                        sims = np.asarray(_NDS.BulkTanimotoSimilarity(fp, _NBANK)); j = int(sims.argmax())
                        lab = _NLL[j] if sims[j] >= 0.7 else None
                    _NCACHE[k] = lab
                return _NCACHE[k]
        for m in list(PC):
            p = PC[m]
            if not p or not p.get('pc') or len(p['pc']) < 2:
                continue
            NP_STATS['molecules'] += 1
            n = len(p['pc'])
            f = _nz(-np.arange(n, dtype=np.float64))
            term = np.zeros(n)
            if NP_MU_LIKE > 0:
                sc = []
                for s in p['pc']:
                    mm = _NC.MolFromSmiles(s)
                    try: sc.append(_NPS.scoreMol(mm, _NPM) if mm is not None else np.nan)
                    except Exception: sc.append(np.nan)
                term += NP_MU_LIKE * _nz(sc)
            if NP_MU_CLS > 0 and m in BASE:
                blab = [_nlab(s, k) for s, k in zip(BASE[m][0][:5], BASE[m][1][:5])]
                NP_STATS['base_labelled'] += int(any(l is not None for l in blab))
                pairs = [(l, 1.0 / (r + 1)) for r, l in enumerate(blab) if l is not None]
                tot = sum(w for _, w in pairs); wt = {}
                for l, w in pairs:
                    wt[l] = wt.get(l, 0.0) + w / tot
                clab = [_nlab(s, k) for s, k in zip(p['pc'], p['pc_keys'])]
                NP_STATS['labelled_cands'] += sum(l is not None for l in clab); NP_STATS['cands'] += n
                term += NP_MU_CLS * _nz([wt.get(l, 0.0) if l is not None else 0.0 for l in clab])
            if not np.any(term):
                continue
            _NTERMS.extend(term.tolist())
            o = [int(i) for i in np.argsort(-(f + term), kind='stable')]
            if o != list(range(n)):
                NP_STATS['reordered'] += 1; NP_STATS['top1_changed'] += int(o[0] != 0)
                fz0 = p['pc_fz'][0]   # the rel merge gate stays on the pure spectral top score
                p['pc'] = [p['pc'][i] for i in o]; p['pc_keys'] = [p['pc_keys'][i] for i in o]
                p['pc_fz'] = [fz0] + [p['pc_fz'][i] for i in o][1:]
    except Exception as _e:
        print('TRACK4b NP signals FAILED -> no-op from here on', repr(_e))
_tm = np.asarray(_NTERMS or [0.0])
print('TRACK4b NP signals | MU_LIKE', NP_MU_LIKE, 'MU_CLS', NP_MU_CLS, '|', NP_STATS,
      f'| term min/median/p99 {_tm.min():.3f}/{np.median(_tm):.3f}/{np.percentile(_tm, 99):.3f}')
"""


def patch_np(t, a):
    if not (a.nplike or a.classcons):
        return t
    block = NP_BLOCK.replace("@MU_LIKE@", repr(float(a.nplike or 0.0))).replace("@MU_CLS@", repr(float(a.classcons or 0.0)))
    return must_replace(t, "print('ICE rerank stats', ice_stats,", block + "print('ICE rerank stats', ice_stats,",
                        "np signals (after the PubChem ICE/GL re-order)")


# ---------------------------------------------------------------- fusion cell (C10)
FUSE_HELPERS = """# === TRACK 4 (patch_supernext.py) ===
MU_F, C1_FREEZE = {mu_f!r}, {c1!r}
_LIBMAX = {{str(k): float(v[2]) for k, v in BASE.items()}}
_RAWTOP = {{str(k): v for k, v in RAW_TOP1.items() if v}}
def _pop_fuse(fsm, fk, fsc):
    if MU_F <= 0 or not POPK or len(fk) < 2:
        return fsm, fk, fsc, 0
    p = np.array([POPK.get(k, np.nan) for k in fk], np.float64)
    if np.all(np.isnan(p)):
        return fsm, fk, fsc, 0
    p = np.where(np.isnan(p), np.nanmedian(p), p)
    s = np.asarray(fsc, np.float64); f = (s - s.mean()) / (s.std() + 1e-9) + MU_F * p
    o = np.argsort(-f, kind='stable')
    return [fsm[i] for i in o], [fk[i] for i in o], [float(f[i]) for i in o], int(o[0] != 0)

if ENG:"""

FREEZE = """        if C1_FREEZE and vs and _LIBMAX.get(str(mid), 0.0) >= LIB_TAU and str(mid) in _RAWTOP:
            _t = _RAWTOP[str(mid)]; _tk = chem.score_key(_t) or _t
            if (chem.score_key(vs[0]) or vs[0]) != _tk:
                vs = ([_t] + [x for x in vs if (chem.score_key(x) or x) != _tk])[:25]; n_frz += 1
        out.append((mid, ';'.join(vs[:25]) if vs else 'CCO'))
"""


def patch_fuse(t, a):
    mu_f = float(a.pop_fused) if a.pop_fused is not None else 0.0
    t = must_replace(t, "\nif ENG:", "\n" + FUSE_HELPERS.format(mu_f=mu_f, c1=bool(a.c1_freeze)), "fusion helpers")
    t = must_replace(t, "    changed = n_ice = 0; out = []", "    changed = n_ice = n_popf = n_frz = 0; out = []", "counters")
    t = must_replace(t, "            fsm, fk, fsc = fuse2(vs, e['smiles'], e['keys'])\n",
                     "            fsm, fk, fsc = fuse2(vs, e['smiles'], e['keys'])\n"
                     "            fsm, fk, fsc, _pm = _pop_fuse(fsm, fk, fsc); n_popf += _pm\n", "pop fused")
    t = must_replace(t, "        out.append((mid, ';'.join(vs[:25]) if vs else 'CCO'))\n", FREEZE, "c1 freeze")
    t = must_replace(t, "f'{time.time()-T0:.0f}s')",
                     "f'{time.time()-T0:.0f}s')\n    print('TRACK4 MU_F', MU_F, 'pop-fused top-1 moved', n_popf, '| C1_FREEZE', C1_FREEZE, 'c1 frozen', n_frz)",
                     "fusion print")
    return t


def tags(a):
    out = []
    if a.pop_precut: out.append("a1precut")
    if a.pop_fused is not None: out.append(f"a2popfused{a.pop_fused:g}")
    if a.mu is not None: out.append(f"mu{a.mu:g}")
    if a.c1_freeze: out.append("a4c1freeze")
    if a.gen_only: out.append("a5genonly")
    if a.nplike: out.append(f"nplike{a.nplike:g}")
    if a.classcons: out.append(f"classcons{a.classcons:g}")
    return out


def build(a):
    nb = json.load(open(a.src, encoding="utf-8"))
    i_cfg = find(nb, lambda s: "POP_LAM, POP_UNION, POOLPOP_MU" in s, "config cell")
    i_base = find(nb, lambda s: "BASE, n_err = {}, 0" in s, "BASE cell")
    i_fuse = find(nb, lambda s: "def fuse2" in s, "fusion cell")
    i_ice = find(nb, lambda s: "print('ICE rerank stats', ice_stats," in s, "ICE/GL cell")
    for i, fn in ((i_cfg, patch_config), (i_base, patch_base), (i_ice, patch_np), (i_fuse, patch_fuse)):
        set_src(nb["cells"][i], fn(src_of(nb["cells"][i]), a))
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
        c.get("metadata", {}).pop("papermill", None)
    return nb


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--pop-precut", action="store_true")
    p.add_argument("--pop-fused", type=float, nargs="?", const=0.15, default=None)
    p.add_argument("--mu", type=float, default=None)
    p.add_argument("--c1-freeze", action="store_true")
    p.add_argument("--gen-only", action="store_true")
    p.add_argument("--nplike", type=float, default=None)
    p.add_argument("--classcons", type=float, default=None)
    p.add_argument("--src", default=SRC)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    if a.gen_only and a.mu is not None:
        p.error("--gen-only sets POOLPOP_MU = 0; do not combine with --mu")
    tg = tags(a)
    pre = "casmi26-t4b" if (a.nplike or a.classcons) else "casmi26-t4"
    dst = a.out or os.path.join(ROOT, f"{pre}-{'-'.join(tg) or 'base'}.ipynb")
    nb = build(a)
    json.dump(nb, open(dst, "w", encoding="utf-8"), indent=1)
    print("wrote", os.path.relpath(dst, ROOT), "| flags:", ", ".join(tg) or "none (instrumentation only)")
    return dst


if __name__ == "__main__":
    main()
