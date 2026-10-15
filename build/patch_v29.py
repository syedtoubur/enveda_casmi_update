"""Track 4 v2 probe generator: patch the 0.411 v29 notebook, one flag per LB probe.

Never hand-edit the generated .ipynb -- change this script and regenerate. The source notebook
is read, never written. Outputs are cleared so every variant uploads cleanly.

Run mode: 0.411 hard-codes IS_RERUN = True, so a "Save Version" commit runs the full ~4.4 h. Every
variant restores the 0.399/0.405 switch: commits (dummy test, known signature) score 12 molecules
in ~20 min; the scoring rerun (KAGGLE_IS_COMPETITION_RERUN, or any other test signature) runs all.

Always applied (no score effect): `TRACK4v2 pop stats` print of the applied pool prior term,
and the `TRACK4v2 MU_ENG ...` counter line in the fusion cell.

Flags (specs/track4_priors_probes/plan_v2.md):
  --mu X          P1  POOLPOP_MU (base 0.15; 0.409 gained +0.005 from 0.15 -> 0.25)
  --pop-eng X     P2  re-sort the engine-A list by z(-rank) + X*pop before fuse2 (pool key -> pop;
                      keys without a pool record get the list median)
  --np-bonus G    P3  pop += G * [LOTUS or NPAtlas flag] in the pool prior (pool_np.npy) and the
                      PubChem prior (pc_np.npy, via V17_PATCH); no-op with a printed notice if the
                      attached popularity dataset has no np arrays
  --pop-precut    P4  apply the pool prior to the ranker's top 200, then cut to TOPN
  --lib-gate      P6  rider: no ICE/GL rerank for molecules with lib_max >= LIB_TAU (C8 base + C10 fused)
  --np-expand     P3e add the LOTUS/NPAtlas structures missing from the public pool
                      (natural_products.parquet, in_casmi_pool False) to engine 2's pool, next to BIO;
                      prints NPX counters and a dMRR upper bound in the fusion cell

Source: --src casmi26-v29-fusion-pop-glmh-0-415.ipynb (0.415, code-identical to 0.411 but already
carrying the commit/rerun switch, so the run-mode patch is a no-op there).

usage: python build/patch_v29.py [flags] [--src in.ipynb] [--out out.ipynb]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from patch_supernext import find, must_replace, set_src, src_of  # noqa: E402

SRC = os.path.join(ROOT, "casmi26-v29-glacier-m-h-full-fusion-0-411.ipynb")


# ---------------------------------------------------------------- run-mode cell (C2)
RUNMODE_OLD = "IS_RERUN = True  # Full 400-molecule execution; never emit the 12-molecule smoke submission\n"
RUNMODE_NEW = ("IS_RERUN = bool(os.getenv('KAGGLE_IS_COMPETITION_RERUN')) or _sig != '654e030e4173780e945252b4a5adf70b'"
               "   # TRACK 4 v2: 0.399/0.405 switch\n")


def patch_runmode(t, a):
    if RUNMODE_OLD not in t and "KAGGLE_IS_COMPETITION_RERUN" in t:
        return t                       # 0.415 source already carries the commit/rerun switch
    return must_replace(t, RUNMODE_OLD, RUNMODE_NEW, "IS_RERUN switch")


# ---------------------------------------------------------------- engine-2 cell (C3)
# P3e: the LOTUS/NPAtlas structures missing from the public pool join engine 2's candidate pool
# next to ChEBI/LIPID MAPS (BIO), fingerprinted by build_pool's own fp_and_mass pass.
_NPX_OLD = '    tr = pd.DataFrame({"ik": L["ik"], "smi": L["smi"]}).dropna().drop_duplicates("ik")\n'
_NPX_NEW = _NPX_OLD + (
    "    try:   # TRACK 4 P3e (patch_v29.py): NP pool expansion\n"
    "        npt = pd.read_parquet(find(\"natural_products.parquet\"), columns=[\"ik14\", \"scorer_key\", \"smiles\", \"mono_mass\", \"in_casmi_pool\"])\n"
    "        npt = npt[~npt.in_casmi_pool.astype(bool) & npt.mono_mass.between(150.0, 1250.0)].dropna(subset=[\"scorer_key\", \"smiles\"])\n"
    "        have = set(co_keys) | set(tr.ik)\n"
    "        npt = npt[~npt.scorer_key.isin(have) & ~npt.ik14.isin(have)].drop_duplicates(\"scorer_key\")\n"
    "        tr = pd.concat([tr, pd.DataFrame({\"ik\": npt.scorer_key.values, \"smi\": npt.smiles.values})], ignore_index=True)\n"
    "        open(os.path.join(os.getcwd(), \"npx_keys.txt\"), \"w\").write(\"\\n\".join(npt.scorer_key.astype(str)))\n"
    "        log(f\"TRACK4 NP expansion +{len(npt):,} structures\")\n"
    "    except Exception as _e:\n"
    "        log(f\"TRACK4 NP expansion failed -> base pool: {_e!r}\")\n"
)


def patch_engine(t, a):
    if not a.np_expand:
        return t
    block = (
        "# === TRACK 4 P3e (patch_v29.py): NP pool expansion inside engine 2's build_pool ===\n"
        f"_npx_old, _npx_new = {_NPX_OLD!r}, {_NPX_NEW!r}\n"
        "assert ENG_FILES['casmi_engine.py'].count(_npx_old) == 1, 'NPX anchor'\n"
        "ENG_FILES['casmi_engine.py'] = ENG_FILES['casmi_engine.py'].replace(_npx_old, _npx_new)\n"
    )
    return must_replace(t, "for _n, _src in ENG_FILES.items():\n", block + "for _n, _src in ENG_FILES.items():\n",
                        "np expand (engine)")


# ---------------------------------------------------------------- PubChem cell (C4)
_V17_OLD1 = "        _W['lp'] = np.load(os.path.join(_POP_DIR, 'pc_lpmid.npy'), mmap_mode='r')\n"
_V17_NEW1 = _V17_OLD1 + ("        try:\n"
                         "            _W['npf'] = np.load(os.path.join(_POP_DIR, 'pc_np.npy'), mmap_mode='r')\n"
                         "        except Exception:\n"
                         "            pass\n")
_V17_OLD2 = ("    pop = (np.asarray(_W['ls'][a0:a0 + n], np.float32) + np.asarray(_W['lp'][a0:a0 + n], np.float32))"
             " if use_pop else np.zeros(n, np.float32)\n")
_V17_NEW2 = _V17_OLD2 + ("    if use_pop and 'npf' in _W and float(os.environ.get('CASMI_NP_BONUS', '0')) > 0:\n"
                         "        pop = pop + float(os.environ['CASMI_NP_BONUS']) * "
                         "((np.asarray(_W['npf'][a0:a0 + n]) & 6) > 0).astype(np.float32)\n")


def patch_pubchem(t, a):
    if not a.np_bonus:
        return t
    block = (
        "# === TRACK 4 v2 (patch_v29.py): natural-product bonus in the PubChem prior ===\n"
        f"os.environ['CASMI_NP_BONUS'] = {str(a.np_bonus)!r}\n"
        "_np_ok = os.path.exists(os.path.join(os.environ.get('CASMI_POP_DIR', ''), 'pc_np.npy'))\n"
        "print('TRACK4v2 np flags (pc_np.npy):', 'found' if _np_ok else 'absent -> PubChem NP bonus is a no-op')\n"
        f"for _old, _new in [({_V17_OLD1!r}, {_V17_NEW1!r}),\n"
        f"                   ({_V17_OLD2!r}, {_V17_NEW2!r})]:\n"
        "    assert V17_PATCH.count(_old) == 1, ('V17 anchor', _old[:60])\n"
        "    V17_PATCH = V17_PATCH.replace(_old, _new)\n"
    )
    return must_replace(t, "# PubChem-only channel in its own process",
                        block + "# PubChem-only channel in its own process", "np bonus (pubchem)")


# ---------------------------------------------------------------- BASE cell (C7)
def patch_base(t, a):
    if a.mu is not None:
        t = must_replace(t, "POOLPOP_MU = 0.15\n", f"POOLPOP_MU = {a.mu!r}\n", "POOLPOP_MU")
    anchor = "assert len(POOL_POP) == len(P.key), (len(POOL_POP), len(P.key))\n"
    add = ""
    if a.np_bonus:
        add += (
            f"NP_BONUS = {a.np_bonus!r}   # TRACK 4 v2: LOTUS (2) or NPAtlas (4) flag; COCONUT (1) is the whole pool\n"
            "try:\n"
            "    _pnp = np.load(find('pool_np.npy'))\n"
            "    assert len(_pnp) == len(POOL_POP), (len(_pnp), len(POOL_POP))\n"
            "    _isnp = (_pnp & 6) > 0\n"
            "    POOL_POP = POOL_POP + NP_BONUS * _isnp\n"
            "    print('TRACK4v2 np flags (pool_np.npy): found | NP-flagged pool rows', int(_isnp.sum()), 'of', len(_isnp))\n"
            "except Exception as _e:\n"
            "    print('TRACK4v2 np flags (pool_np.npy): absent -> pool NP bonus is a no-op', repr(_e))\n"
        )
    add += ("_pp = POOLPOP_MU * POOL_POP\n"
            "print('TRACK4v2 pop stats (mu*pop) min/median/p99/max', round(float(_pp.min()), 3), round(float(np.median(_pp)), 3),"
            " round(float(np.percentile(_pp, 99)), 3), round(float(_pp.max()), 3))\n")
    t = must_replace(t, anchor, anchor + add, "pop stats")
    if a.pop_precut:
        t = must_replace(t, "n_pp = 0\n",
                         "n_pp = 0\nTOPN_PRE, N_PROMOTED = (max(TOPN, 200) if POOLPOP_MU > 0 else TOPN), 0   # TRACK 4 v2: prior before the cut\n",
                         "TOPN_PRE")
        t = must_replace(t, "                    if len(smis) >= TOPN: break\n",
                         "                    if len(smis) >= TOPN_PRE: break\n", "loop cut")
        t = must_replace(t, "    if POOLPOP_MU > 0 and len(scs) > 1:",
                         "    _raw_top = set(keys[:TOPN])\n    if POOLPOP_MU > 0 and len(scs) > 1:", "raw top")
        t = must_replace(t, "    BASE[mid] = (smis, keys, lib_max, scs, forms)\n",
                         "    smis, keys, scs, forms = smis[:TOPN], keys[:TOPN], scs[:TOPN], forms[:TOPN]\n"
                         "    N_PROMOTED += sum(k not in _raw_top for k in keys)\n"
                         "    BASE[mid] = (smis, keys, lib_max, scs, forms)\n", "cut back to TOPN")
        t = must_replace(t, "print('errors', n_err, '| popularity re-ranked lists', n_pp)",
                         "print('errors', n_err, '| popularity re-ranked lists', n_pp)\n"
                         "print('TRACK4v2 TOPN_PRE', TOPN_PRE, '| promoted into top', TOPN, ':', N_PROMOTED)", "precut print")
    return t


# ---------------------------------------------------------------- ICE/GL cell (C8)
def patch_ice(t, a):
    if not a.lib_gate:
        return t
    t = must_replace(t, "BASE_ICE = True ", "LIB_GATE = True   # TRACK 4 v2: no forward rerank for library hits\nBASE_ICE = True ",
                     "LIB_GATE")
    t = must_replace(t, "    if BASE_ICE and ice and any(v is not None for v in ice.values()):\n",
                     "    if BASE_ICE and ice and any(v is not None for v in ice.values()) and not (LIB_GATE and lib_max >= LIB_TAU):\n",
                     "base lib gate")
    t = must_replace(t, "print('GL rerank stats', gl_stats, f'{time.time()-T0:.0f}s')",
                     "print('GL rerank stats', gl_stats, f'{time.time()-T0:.0f}s')\n"
                     "print('TRACK4v2 LIB_GATE base lists kept out of ICE/GL:', sum(v[2] >= LIB_TAU for v in BASE.values()))",
                     "lib gate print")
    return t


# ---------------------------------------------------------------- fusion cell (C10)
HELPERS = """ALPHA, KRR = 0.6, 3.0
# === TRACK 4 v2 (patch_v29.py) ===
MU_ENG, LIB_GATE_F = @MU_ENG@, @LIB_GATE@
_LIBMAX = {}
for _k, _v in BASE.items():
    try:
        _LIBMAX[str(_k)] = float(_v[2])
    except Exception:
        pass
_KPOP = None
if MU_ENG > 0:
    try:
        _KPOP = {(k.decode() if isinstance(k, bytes) else str(k)): float(v) for k, v in zip(P.key, POOL_POP)}
        print('TRACK4v2 engine pop map', len(_KPOP), 'keys, e.g.', next(iter(_KPOP), None))
    except Exception as _e:
        print('TRACK4v2 engine pop map failed', repr(_e))
N_ENGPOP = N_ENGPOP_T1 = N_ENGPOP_HIT = N_ENGPOP_ALL = N_LIBG = 0
def _pop_eng(es, ek):
    # engine-A list: z(-rank) + MU_ENG * pop; keys without a pool record get the list median
    global N_ENGPOP, N_ENGPOP_T1, N_ENGPOP_HIT, N_ENGPOP_ALL
    try:
        if not _KPOP or len(es) < 2:
            return es, ek
        pv = np.array([_KPOP.get(str(k), np.nan) for k in ek], np.float64)
        hit = np.isfinite(pv); N_ENGPOP_HIT += int(hit.sum()); N_ENGPOP_ALL += len(pv)
        if not hit.any():
            return es, ek
        pv[~hit] = np.median(pv[hit])
        r = -np.arange(1, len(es) + 1, dtype=np.float64)
        o = np.argsort(-((r - r.mean()) / (r.std() + 1e-9) + MU_ENG * pv), kind='stable')
        N_ENGPOP += 1; N_ENGPOP_T1 += int(o[0] != 0)
        return [es[i] for i in o], [ek[i] for i in o]
    except Exception as _e:
        print('TRACK4v2 engine pop failed', repr(_e))
        return es, ek
"""


def patch_fuse(t, a):
    h = HELPERS.replace("@MU_ENG@", repr(float(a.pop_eng or 0.0))).replace("@LIB_GATE@", repr(bool(a.lib_gate)))
    t = must_replace(t, "ALPHA, KRR = 0.6, 3.0\n", h, "helpers")
    t = must_replace(t, "            fsm, fk, fsc = fuse2(vs, e['smiles'], e['keys'])\n",
                     "            _es, _ek = _pop_eng(list(e['smiles']), list(e['keys'])) if MU_ENG > 0 else (e['smiles'], e['keys'])\n"
                     "            fsm, fk, fsc = fuse2(vs, _es, _ek)\n", "engine pop")
    t = must_replace(t, "            if ice:\n",
                     "            _gated = LIB_GATE_F and _LIBMAX.get(str(mid), 0.0) >= LIB_TAU\n"
                     "            N_LIBG += int(bool(ice) and _gated)\n"
                     "            if ice and not _gated:\n", "fused lib gate")
    t = must_replace(t, "f'{time.time()-T0:.0f}s')\n",
                     "f'{time.time()-T0:.0f}s')\n"
                     "    print('TRACK4v2 MU_ENG', MU_ENG, 'engine lists re-sorted', N_ENGPOP, 'top-1 moved', N_ENGPOP_T1,"
                     " '| pop key hits', N_ENGPOP_HIT, 'of', N_ENGPOP_ALL, '| LIB_GATE', LIB_GATE_F, 'fused lists not reranked', N_LIBG)\n",
                     "fusion print")
    if a.np_expand:
        t = must_replace(t, "ALPHA, KRR = 0.6, 3.0\n", "ALPHA, KRR = 0.6, 3.0\n" + NPX_COUNT, "npx helpers")
        t = must_replace(t, "            fsm, fk, fsc = fuse2(vs, _es, _ek)\n",
                         "            fsm, fk, fsc = fuse2(vs, _es, _ek)\n            _k_of = dict(zip(fsm, fk))\n", "npx key map")
        t = must_replace(t, "            vs = fsm[:25]\n",
                         "            vs = fsm[:25]\n            _npx_count(e['keys'], [_k_of.get(x) for x in vs])\n", "npx count")
        t = must_replace(t, "    fs.to_csv('submission.csv', index=False)\n",
                         "    fs.to_csv('submission.csv', index=False)\n"
                         "    print('TRACK4 NPX keys', len(_NPX), '| in engine top-40: molecules', N_NPX_ENG, '| in final top-25', N_NPX_TOP,"
                         " '| at rank 1', N_NPX_R1, '| dMRR bound', round(NPX_BOUND / max(len(v4), 1), 4), '| ENG duplicate keys', N_ENG_DUP)\n",
                         "npx print")
    return t


NPX_COUNT = """# === TRACK 4 P3e (patch_v29.py): NP-expansion counters ===
try:
    _NPX = set(l.strip() for l in open(os.path.join(ENG_DIR, 'npx_keys.txt')) if l.strip())
except Exception as _e:
    _NPX = set(); print('TRACK4 NPX keys absent -> counters stay 0', repr(_e))
N_NPX_ENG = N_NPX_TOP = N_NPX_R1 = N_ENG_DUP = 0; NPX_BOUND = 0.0
def _npx_count(ek, fk25):
    global N_NPX_ENG, N_NPX_TOP, N_NPX_R1, N_ENG_DUP, NPX_BOUND
    try:   # counters only: never allowed to break the fusion loop
        ek = [k for k in ek if k]; N_ENG_DUP += len(ek) - len(set(ek))
        N_NPX_ENG += int(any(k in _NPX for k in ek))
        r = [i for i, k in enumerate(fk25, 1) if k in _NPX]
        if r:
            N_NPX_TOP += 1; N_NPX_R1 += int(r[0] == 1); NPX_BOUND += 1.0 / r[0]
    except Exception:
        pass
"""


def tags(a):
    out = []
    if a.mu is not None: out.append(f"mu{a.mu:g}")
    if a.pop_eng: out.append(f"popeng{a.pop_eng:g}")
    if a.np_bonus: out.append(f"np{a.np_bonus:g}")
    if a.pop_precut: out.append("precut")
    if a.lib_gate: out.append("libgate")
    if a.np_expand: out.append("npx")
    return out


def build(a):
    nb = json.load(open(a.src, encoding="utf-8"))
    cells = [(find(nb, lambda s: "IS_RERUN =" in s, "run-mode cell"), patch_runmode),
             (find(nb, lambda s: "ENG_FILES = {" in s, "engine cell"), patch_engine),
             (find(nb, lambda s: s.startswith("V17_PATCH = "), "PubChem cell"), patch_pubchem),
             (find(nb, lambda s: "POOLPOP_MU = 0.15" in s, "BASE cell"), patch_base),
             (find(nb, lambda s: "BASE_ICE = True" in s, "ICE cell"), patch_ice),
             (find(nb, lambda s: "def fuse2" in s, "fusion cell"), patch_fuse)]
    for i, fn in cells:
        set_src(nb["cells"][i], fn(src_of(nb["cells"][i]), a))
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
        c.get("metadata", {}).pop("papermill", None)
    return nb


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--mu", type=float, default=None)
    p.add_argument("--pop-eng", type=float, default=None)
    p.add_argument("--np-bonus", type=float, default=None)
    p.add_argument("--pop-precut", action="store_true")
    p.add_argument("--lib-gate", action="store_true")
    p.add_argument("--np-expand", action="store_true")
    p.add_argument("--src", default=SRC)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    tg = tags(a)
    dst = a.out or os.path.join(ROOT, f"casmi26-t4v2-{'-'.join(tg) or 'base'}.ipynb")
    nb = build(a)
    json.dump(nb, open(dst, "w", encoding="utf-8"), indent=1)
    print("wrote", os.path.abspath(dst), "| flags:", ", ".join(tg) or "none (instrumentation only)")
    return dst


if __name__ == "__main__":
    main()
