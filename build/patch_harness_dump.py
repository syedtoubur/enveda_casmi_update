"""Harness instrument: casmi-notebook-0-433 + post-scoring state dump + in-run replay equivalence + val mode.

Plan: ~/.claude/plans/here-i-am-trying-wiggly-pine.md (P1). Base: casmi-notebook-0-433.ipynb, read only.
The notebook's own cells are NOT changed, so its submission.csv is the base's. Three additions:
  1. after the commit switch: VAL_SHARD. '' = normal run. 'C2' / 'C1_npx' = val mode: test = that val_key cohort
     (val.parquet), train = train minus the cohort's ban_rows, every molecule scored with the full ICE budget.
  2. inside the simulator cell, just before the C8 rerank loop: snapshot BASE and PC (deep copies).
  3. a final cell: casmi_post.run(snapshot) must reproduce submission.csv for every molecule
     (`POST EQUIV ok n of n | mismatches 0`, else AssertionError), then post_dump/ = state.pkl + the fuse.py,
     gl_fuse.py and casmi/*.py sources the replay needs.
Never hand-edit the generated .ipynb; change this script (or casmi_post.py) and regenerate.

usage: python build/patch_harness_dump.py [--shard C2|C1_npx|''] [--src in.ipynb] [--out out.ipynb]
"""
from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "casmi-notebook-0-433.ipynb")
SHARDS = ("", "C2", "C1_npx", "C2_np", "C1_np_fp0", "C2_hard", "C1_xlib", "pseudo_c3")   # C2_hard / C1_xlib: scorecard battery (2026-10-13)
NP_SHARDS = ("C2_np", "C1_np_fp0")       # Option A: BASE bank = fpnet_0, Engine 2 ranker rows without np queries
C3_SHARD = "pseudo_c3"                   # Class 3 (campaign 20261015-c3-gen-slot2): truth banned from library AND every pool


def src_of(cell):
    return "".join(cell["source"])


def set_src(cell, text):
    cell["source"] = text.splitlines(keepends=True)


def must_replace(text, old, new, what):
    if text.count(old) != 1:
        raise SystemExit(f"PATCH FAILED: anchor for {what} occurs {text.count(old)} times:\n  {old[:120]}")
    return text.replace(old, new)


def find(nb, pred, what):
    hits = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code" and pred(src_of(c))]
    if len(hits) != 1:
        raise SystemExit(f"PATCH FAILED: {what} matched {len(hits)} cells")
    return hits[0]


# The 0.433 ICEBERG preseed only looks for a cp313 rdkit-2025.3.6 wheel. Dump kernels run on the control's pinned
# image (E006, Python 3.12), so ICE died at its health assert and state["ice"] came back empty (handoff
# 20261014-0100). Pick the wheel by the interpreter's own tags, as E006's simulator_rdkit_wheel does: the bundled
# cp312 `.whl.ice` on 3.12, the mounted cp313 wheel on 3.13.
PRESEED_OLD = (
    "    # ...and add the cp313 build of the SAME rdkit release\n"
    "    new = sorted(_g.glob('/kaggle/input/**/rdkit-2025.3.6-cp313*.whl', recursive=True))\n"
    "    if not new:\n"
    "        new = [w for w in _g.glob('/kaggle/input/**/rdkit-*.whl', recursive=True) if 'cp313' in w]\n"
    "\n"
    "    if not new:\n"
    "        raise FileNotFoundError('cp313 rdkit wheel not found; dataset not attached?')\n")
PRESEED_NEW = (
    "    # HARNESS (patch_harness_dump.py): the rdkit 2025.3.6 build whose tags this interpreter supports (E006 rule)\n"
    "    from pip._vendor.packaging.tags import sys_tags as _st\n"
    "    from pip._vendor.packaging.utils import parse_wheel_filename as _pw\n"
    "    _ok = set(_st())\n"
    "    _cands = (sorted(_g.glob(_o.path.join(pkg, 'wheels', 'rdkit-2025.3.6-*.whl*'))) +\n"
    "              sorted(_g.glob('/kaggle/input/**/rdkit-2025.3.6-*.whl', recursive=True)))\n"
    "    new = [w for w in _cands if _ok.intersection(_pw(_o.path.basename(w).removesuffix('.ice'))[3])]\n"
    "    print('[harness] simulator rdkit wheel:', new[:1], 'python', _sys.version.split()[0], flush=True)\n"
    "    if not new:\n"
    "        raise FileNotFoundError(f'no rdkit 2025.3.6 wheel for this interpreter; candidates: {_cands}')\n")


# lib-isomer-margin C7 graft (both notebooks; the cell that holds `BASE, n_err = {}, 0` and the C7 loop). Per molecule,
# LIB_SIM[mid] = {SMILES: best reference similarity} for the BASE top-10 candidates that have a library reference.
# `_lc` indexes X by the engine's FEATURES order (casmi.engine.py:66-81: X[:, COL[name]] with COL = enumerate(FEATURES)).
LIB_SIM_INIT_OLD = "BASE, n_err = {}, 0\n"
LIB_SIM_INIT_NEW = ("BASE, n_err = {}, 0\n"
                    "LIB_SIM = {}   # HARNESS (lib-isomer-margin): per-molecule library similarity for the post knob\n")
LIB_SIM_LOOP_OLD = ("                    if len(smis) >= TOPN: break\n"
                    "        except Exception as e:\n"
                    "            print('ERROR', mid, repr(e)); n_err += 1\n")
LIB_SIM_LOOP_NEW = ("                    if len(smis) >= TOPN: break\n"
                    "                # HARNESS (lib-isomer-margin): best reference similarity of the BASE top-10\n"
                    "                _lc = list(FEATURES).index('lib_max')\n"
                    "                LIB_SIM[mid] = {C.smiles[i]: float(X[i, _lc]) for i in order[:10] if X[i, _lc] > 0}\n"
                    "        except Exception as e:\n"
                    "            print('ERROR', mid, repr(e)); n_err += 1\n")


def lib_sim_graft(nb):
    """Adds the C7 graft to the cell that builds BASE (exactly one such cell, both anchors checked once)."""
    i = find(nb, lambda s: "lib_max = float(info.get('lib_max', 0.0))" in s and LIB_SIM_LOOP_OLD in s, "C7 cell")
    text = src_of(nb["cells"][i])
    text = must_replace(text, LIB_SIM_INIT_OLD, LIB_SIM_INIT_NEW, "LIB_SIM init")
    text = must_replace(text, LIB_SIM_LOOP_OLD, LIB_SIM_LOOP_NEW, "LIB_SIM graft")
    set_src(nb["cells"][i], text)
    return nb


def code_cell(text):
    c = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": []}
    set_src(c, text)
    return c


VAL_CELL = r'''# === HARNESS (patch_harness_dump.py): val mode. '' = the normal run ===
import os, pickle                     # self-contained: this cell runs above the base's own imports (R07)
import numpy as np, pandas as pd
VAL_SHARD = @SHARD@
def _hx_val_records(recs, shard):
    """val_key records of a shard. Option A: C2_np = the C1_npx molecules (the 245 enveda-np-examples, which fpnet_0
    never trained on) with every train row of the truth banned by key (Class 2); C1_np_fp0 = C1_npx as is."""
    src = {'C2_np': 'C1_npx', 'C1_np_fp0': 'C1_npx'}.get(shard, shard)
    out = [dict(r) for r in recs if r['cohort'] == src]
    if shard == 'C2_np':
        out = [dict(r, cohort='C2_np', cls=2, ban_keys=[r['inchikey14']]) for r in out]
    return out
# end _hx_val_records
if VAL_SHARD and VAL_SHARD != 'pseudo_c3':
    _vk = _hx_val_records(pickle.load(open(find('val_key.pkl'), 'rb')), VAL_SHARD)
    _ids = [r['molecule_id'] for r in _vk]
    _vq = pd.read_parquet(find('val.parquet'))
    _vq = _vq[_vq.molecule_id.isin(set(_ids))].reset_index(drop=True)
    _tcols = list(pd.read_parquet(find('test.parquet')).columns)
    for _c in _tcols:
        if _c not in _vq.columns:
            _vq[_c] = None
    _tr = pd.read_parquet(find('train.parquet'))
    _ban = sorted({int(x) for r in _vk for x in r['ban_rows']})
    assert _ban and max(_ban) < len(_tr), (len(_tr), _ban[-1:])
    if 'precursor_mz' in _tr.columns:                                   # ban rows sit at the query mass
        _pm = _tr.precursor_mz.values
        _q = _vq.groupby('molecule_id').precursor_mz.median().to_dict()
        _miss = [r for r in _vk if r['ban_rows'] and r['molecule_id'] in _q
                 and min(abs(_pm[int(x)] - _q[r['molecule_id']]) for x in r['ban_rows']) >= 0.02]
        print(f'VAL ban-row check ok {len(_vk) - len(_miss)} of {len(_vk)}', flush=True)
        if 'adduct' in _tr.columns:                                     # H1b: a miss is explained when no truth
            _qa = _vq.groupby('molecule_id').adduct.agg(set).to_dict()  # row shares the query's adduct
            _expl = [r for r in _miss if not (_qa[r['molecule_id']] & set(_tr.adduct.values[list(map(int, r['ban_rows']))]))]
            print(f'VAL ban-row misses {len(_miss)}: other-adduct explained {len(_expl)}', flush=True)
            for r in [r for r in _miss if r not in _expl][:5]:
                _rows = list(map(int, r['ban_rows']))
                print('  unexplained', r['molecule_id'], sorted(_qa[r['molecule_id']]), round(_q[r['molecule_id']], 4),
                      list(zip(_tr.adduct.values[_rows], np.round(_pm[_rows], 4)))[:6], flush=True)
    else:
        print('VAL ban-row check skipped: train has no precursor_mz; columns', list(_tr.columns)[:20], flush=True)
    _keep = np.ones(len(_tr), bool); _keep[_ban] = False
    _bk = {k for r in _vk for k in (r.get('ban_keys') or [])}           # C2: every row with the truth IK14
    _smi = next((c for c in ('normalized_smiles', 'smiles') if c in _tr.columns), None)
    if _bk and 'inchikey14' in _tr.columns:                             # train schema: native plain IK14 column
        _ik = _tr.inchikey14.astype(str)
    elif _bk and _smi:
        from rdkit import Chem as _Chem, RDLogger as _RDL
        _RDL.DisableLog('rdApp.*')
        def _ik14(s):
            m = _Chem.MolFromSmiles(s) if isinstance(s, str) else None
            return _Chem.MolToInchiKey(m)[:14] if m is not None else ''
        _ikm = {s: _ik14(s) for s in pd.unique(_tr[_smi])}
        _ik = _tr[_smi].map(_ikm)
    else:
        _ik = None
        if _bk:
            print('VAL key-ban skipped: train has no inchikey14/smiles; columns', list(_tr.columns)[:20], flush=True)
    if _ik is not None:
        _hit = _ik.isin(_bk).values
        print(f'VAL ban extra rows by key {int((_hit & _keep).sum())}', flush=True)
        _keep &= ~_hit
        print(f'VAL key-ban leftover truth rows {int(_ik[_keep].isin(_bk).sum())}', flush=True)
    VD = '/kaggle/working/val_comp'; os.makedirs(VD, exist_ok=True)
    _vq[_tcols].to_parquet(os.path.join(VD, 'test.parquet'))
    _tr[_keep].reset_index(drop=True).to_parquet(os.path.join(VD, 'train.parquet'))
    pd.DataFrame({'molecule_id': _ids, 'smiles': 'CCO'}).to_csv(os.path.join(VD, 'sample_submission.csv'), index=False)
    del _tr
    COMP, IS_RERUN, ICE_BUDGET = VD, True, 5400
    print(f'VAL MODE {VAL_SHARD}: molecules {len(_ids)} spectra {len(_vq)} banned train rows {len(_ban)}', flush=True)
'''

SNAP_ANCHOR = "for m in list(BASE):\n    smis, keys, lib_max, scs, forms = BASE[m]\n"
SNAP = r'''# >>> HARNESS (patch_harness_dump.py): post-scoring state before the C8 rerank loop
import copy as _hx_copy, pickle as _hx_pickle, shutil as _hx_shutil   # _hx_ names: the base re-binds _pk (R07)
_POST_SNAP = dict(base=_hx_copy.deepcopy(BASE), pc=_hx_copy.deepcopy(PC))
def _harness_dump(consts):
    """post_dump/ = state.pkl + the sources the replay imports. Written here, before C9-C12 can fail, and
    re-written by the final cell with the C9/C10 constants."""
    _D = '/kaggle/working/post_dump'; os.makedirs(os.path.join(_D, 'src', 'casmi'), exist_ok=True)
    st = dict(base=_POST_SNAP['base'], pc=_POST_SNAP['pc'], eng=ENG, ice=ICE_SCORES, gl=GL_SCORES,
              te_mids=[str(m) for m in te.molecule_id.unique()],
              sample_ids=[str(m) for m in pd.read_csv(os.path.join(COMP, 'sample_submission.csv')).molecule_id],
              targets={str(k): float(v) for k, v in te.groupby('molecule_id').nm.median().items()},
              lib_sim=dict(globals().get('LIB_SIM') or {}),
              consts=consts, shard=VAL_SHARD, rdkit=rdkit.__version__)
    _hx_pickle.dump(st, open(os.path.join(_D, 'state.pkl'), 'wb'), protocol=4)
    for _src, _dst in ((globals().get('ICE_PKG'), 'fuse.py'), (globals().get('GL_PKG'), 'gl_fuse.py')):
        if _src and os.path.exists(os.path.join(_src, _dst)):
            _hx_shutil.copy(os.path.join(_src, _dst), os.path.join(_D, 'src', _dst))
    for _f in glob.glob(os.path.join(V4CODE, 'casmi', '*.py')):
        _hx_shutil.copy(_f, os.path.join(_D, 'src', 'casmi', os.path.basename(_f)))
    print('POST DUMP', _D, f'molecules {len(st["te_mids"])}', sorted(os.listdir(os.path.join(_D, 'src'))), flush=True)
    return st
try:
    _harness_dump(dict(TOPN=TOPN, ICE_LAM=ICE_LAM, GL_LAM=GL_LAM, LIB_TAU=LIB_TAU, REL_TH=REL_TH,
                       SLOTS_AGG=SLOTS_AGG, SLOTS_GENTLE=SLOTS_GENTLE, BASE_ICE=BASE_ICE, ICE_PC=ICE_PC))
except Exception as _e:
    print('POST DUMP FAILED (early):', repr(_e), flush=True)
# <<< HARNESS
'''

FINAL_CELL = r'''# === HARNESS (patch_harness_dump.py): replay equivalence + full post-scoring dump ===
import importlib as _hx_il
open('/kaggle/working/casmi_post.py', 'w').write(POST_SRC)
sys.path.insert(0, '/kaggle/working'); import casmi_post; _hx_il.reload(casmi_post)
_state = _harness_dump(dict(TOPN=TOPN, ICE_LAM=ICE_LAM, GL_LAM=GL_LAM, LIB_TAU=LIB_TAU, REL_TH=REL_TH,
                            SLOTS_AGG=SLOTS_AGG, SLOTS_GENTLE=SLOTS_GENTLE, S_TAU=S_TAU, POP_TAU=POP_TAU,
                            ALPHA=ALPHA, KRR=KRR, BASE_ICE=BASE_ICE, CELL19_ICE=CELL19_ICE, ICE_PC=ICE_PC))
_hx_shutil.copy('/kaggle/working/casmi_post.py', '/kaggle/working/post_dump/src/casmi_post.py')
_deps = dict(score_key=chem.score_key, form=_form2, ice_rerank=ice_fuse.rerank,
             gl_rerank_multi=gl_fuse.rerank_multi,
             dreams=(_dm_mass, _dm_smiles) if _dm_mass is not None else None)
_replay = casmi_post.run(_state, _deps)
_got = pd.read_csv('submission.csv')
_bad = [m for m, s in zip(_got.molecule_id.astype(str), _got.smiles)
        if ';'.join(_replay.get(m, ['<missing>'])) != str(s)]
print(f'POST EQUIV ok {len(_got) - len(_bad)} of {len(_got)} | mismatches {len(_bad)}', flush=True)
assert not _bad, f'casmi_post replay differs from submission.csv for {len(_bad)} molecules, e.g. {_bad[:5]}'
'''


BANK_CELL = r"""# === HARNESS (patch_harness_dump.py): BASE bank -> fpnet_0 on the np shards (Option A; fpnet_0 never saw them) ===
import os
if VAL_SHARD in ('C2_np', 'C1_np_fp0'):
    _hx_f0 = os.path.join(V4MOD, 'fpnet_0.pt')
    _hx_bank = fpnet.ModelBank([_hx_f0], device=dev)
    assert _hx_bank.nbits == 10226 == E.bank.nbits, (_hx_bank.nbits, E.bank.nbits)
    E.bank = bank = _hx_bank
    print('HARNESS fpnet_0 swaps 1', _hx_f0, flush=True)
"""

# Engine 2 runs in a subprocess (eng_runner.py, written by the base cell). A .pth in a site dir imports this hook in
# every new interpreter; it wraps casmi_engine.fit_rankers ("our ranker") to drop the rows of the np-example queries.
HX_HOOK_SRC = r"""# HARNESS (patch_harness_dump.py): Engine 2 ranker-row filter, active only with HX_RANKER_FIELD set
import os, sys
_F, _V = os.environ.get('HX_RANKER_FIELD'), os.environ.get('HX_RANKER_DROP')
_N = os.environ.get('HX_RANKER_NAMES', '')         # optional: _F holds codes into this name table (e.g. Q -> sets)
_B = os.environ.get('HX_BAN_KEYS', '')             # pseudo_c3: json {'pool_ban': [...]}, dropped from build_pool
if (_F and _V is not None) or _B:
    import importlib.abc, importlib.machinery

    def _patch(mod):
        if _B:
            _patch_pool(mod)
        if _F and _V is not None:
            _patch_rank(mod)

    def _patch_pool(mod):
        import json
        _j = json.load(open(_B))
        ban, tm = set(_j['pool_ban']), sorted(_j.get('masses') or [])
        orig_bp = mod.build_pool

        def build_pool(*a, **k):
            import numpy as np
            import pandas as pd
            P = orig_bp(*a, **k)
            plain = pd.Series(P['keys']).isin(ban).values
            # the pool is keyed by plain IK14 but the lists are re-keyed by canon_key (tautomer-canonical): a tautomer
            # of the truth under another plain key leaked into 8 lists (pseudoc3 v2). canon_key on rows near a truth.
            taut = np.zeros(len(plain), bool)
            if tm:
                t, m = np.asarray(tm, np.float64), np.asarray(P['mass'], np.float64)
                j = np.searchsorted(t, m)
                d = np.minimum(np.abs(m - t[np.clip(j - 1, 0, len(t) - 1)]), np.abs(m - t[np.clip(j, 0, len(t) - 1)]))
                idx = np.where((d <= 10e-6 * np.maximum(np.nan_to_num(m), 1.0)) & ~plain)[0]
                smi = [P['smiles'][i] for i in idx]
                if len(smi) > 2000:
                    from multiprocessing import Pool as _MP
                    with _MP(os.cpu_count() or 4) as mp:
                        ck = mp.map(mod.canon_key, smi, chunksize=64)
                else:
                    ck = [mod.canon_key(s) for s in smi]
                taut[idx] = [c in ban for c in ck]
                print(f'HARNESS c3 ban engine2 canon_key on {len(idx)} pool rows near a truth mass', flush=True)
            keep = ~(plain | taut)
            P = dict(P, fp=P['fp'][keep], mass=P['mass'][keep], keys=P['keys'][keep], smiles=P['smiles'][keep])
            P['k2i'] = pd.Series(np.arange(len(P['keys'])), index=P['keys'])
            P['k2i'] = P['k2i'][~P['k2i'].index.duplicated()]
            open(_B + '.eng2', 'w').write(f'{int((~keep).sum())} {len(keep)} {int(plain.sum())} {int(taut.sum())}')
            print(f'HARNESS c3 ban engine2 pool removed {int((~keep).sum())} of {len(keep)} '
                  f'(plain {int(plain.sum())}, tautomer {int(taut.sum())})', flush=True)
            return P
        mod.build_pool = build_pool

    def _patch_rank(mod):
        orig = mod.fit_rankers

        def fit_rankers(path, *a, **k):
            import numpy as np
            z = np.load(path, allow_pickle=True)
            q = z[_F]
            n = len(q)
            if _N:
                q = np.asarray(z[_N])[np.asarray(q).astype(np.int64)]
            drop = np.isin(np.asarray(q).astype(str), _V.split('|'))
            out = {key: (z[key][~drop] if z[key].ndim >= 1 and z[key].shape[0] == n else z[key]) for key in z.files}
            p2 = os.environ.get('HX_RANKER_OUT', '/kaggle/working/hx_rank_rows.npz')
            np.savez(p2, **out)
            open(p2 + '.dropped', 'w').write(f'{int(drop.sum())} {n}')
            print(f'HARNESS ranker rows dropped {int(drop.sum())} of {n}', flush=True)
            return orig(p2, *a, **k)
        mod.fit_rankers = fit_rankers

    class _HxFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name != 'casmi_engine':
                return None
            spec = importlib.machinery.PathFinder.find_spec(name, path)
            if spec is None or spec.loader is None:
                return spec
            _exec = spec.loader.exec_module

            def exec_module(mod, _exec=_exec):
                _exec(mod)
                _patch(mod)
            spec.loader.exec_module = exec_module
            return spec
    sys.meta_path.insert(0, _HxFinder())
"""

RANKER_CELL = r"""# === HARNESS (patch_harness_dump.py): Engine 2 "our ranker" without enveda-np-examples query rows (np shards) ===
import os, sys, subprocess, site as _hx_site
HX_RANKER_FIELD, HX_RANKER_DROP, HX_RANKER_NAMES = @RFIELD@, @RDROP@, @RNAMES@   # verified by the A0 kernel's NPZ printout
HX_RANKER_OUT = '/kaggle/working/hx_rank_rows.npz'
HX_RANKER_ON = VAL_SHARD in ('C2_np', 'C1_np_fp0')
if HX_RANKER_ON:
    os.environ.update(HX_RANKER_FIELD=HX_RANKER_FIELD, HX_RANKER_DROP=HX_RANKER_DROP, HX_RANKER_NAMES=HX_RANKER_NAMES,
                      HX_RANKER_OUT=HX_RANKER_OUT)
if HX_RANKER_ON or os.environ.get('HX_BAN_KEYS'):
    _hx_ok = None
    for _hx_dir in list(_hx_site.getsitepackages()) + [_hx_site.getusersitepackages()]:
        try:
            os.makedirs(_hx_dir, exist_ok=True)
            open(os.path.join(_hx_dir, 'hx_ranker_hook.py'), 'w').write(@HOOK@)
            open(os.path.join(_hx_dir, 'zz_hx_ranker_hook.pth'), 'w').write('import hx_ranker_hook\n')
        except OSError:
            continue
        _hx_r = subprocess.run([sys.executable, '-c', 'import sys; print("hx_ranker_hook" in sys.modules)'],
                               capture_output=True, text=True)
        if _hx_r.stdout.strip() == 'True':
            _hx_ok = _hx_dir
            break
    assert _hx_ok, 'HARNESS ranker hook not loaded by a fresh interpreter'
    print('HARNESS ranker hook installed', _hx_ok, HX_RANKER_ON, HX_RANKER_FIELD, HX_RANKER_NAMES, HX_RANKER_DROP,
          os.environ.get('HX_BAN_KEYS', ''), flush=True)
"""

RANKER_REPORT_CELL = r"""# === HARNESS (patch_harness_dump.py): ranker-row filter report (the Engine 2 log tail may cut its own line) ===
import os
if HX_RANKER_ON:
    _hx_f = HX_RANKER_OUT + '.dropped'
    _hx_d = open(_hx_f).read().split() if os.path.exists(_hx_f) else ['0', '0']
    print(f'HARNESS ranker rows dropped {_hx_d[0]} of {_hx_d[1]}', flush=True)
    assert int(_hx_d[0]) > 0, 'HARNESS ranker-row filter did not fire (see the Engine 2 log above)'
if os.environ.get('HX_BAN_KEYS'):
    _hx_f = os.environ['HX_BAN_KEYS'] + '.eng2'
    _hx_d = (open(_hx_f).read().split() if os.path.exists(_hx_f) else ['-1', '0']) + ['?', '?']
    print(f'HARNESS c3 ban engine2 pool removed {_hx_d[0]} of {_hx_d[1]} (plain {_hx_d[2]}, tautomer {_hx_d[3]})',
          flush=True)
    assert int(_hx_d[0]) > 0, 'HARNESS c3 Engine 2 pool ban did not fire (see the Engine 2 log above)'
"""

ENG_GUARD_CELL = r"""# === HARNESS (patch_harness_dump.py): Engine 2 coverage guard, val mode only (the next base cell asserts) ===
# Missing molecules get empty lists: C10 / C12 / casmi_post treat an empty list exactly like a missing one.
if VAL_SHARD:
    _hx_act = [str(m) for m in te.molecule_id.unique()]
    _hx_miss = sorted(m for m in _hx_act if m not in set(map(str, ENG)))
    _hx_cov = 1.0 - len(_hx_miss) / max(1, len(_hx_act))
    assert _hx_cov >= 0.95, f'HARNESS ENG coverage {_hx_cov:.3f} < 0.95, missing e.g. {_hx_miss[:10]}'
    for _hx_m in _hx_miss:
        ENG[_hx_m] = dict(smiles=[], keys=[])
    print(f'HARNESS ENG fill {len(_hx_miss)} {_hx_miss}', flush=True)
"""


C3_VAL_CELL = r'''# === HARNESS (patch_harness_dump.py): pseudo_c3 val mode (Class 3: truth banned from library AND every pool) ===
import os, sys, json                     # self-contained: this cell runs above the base's own imports (R07)
import numpy as np, pandas as pd
HX_C3_BAN, HX_PID = set(), {}            # HX_PID[mid] = {key: Engine 1 pid} (-1 = generated), filled in the C7 cell
if VAL_SHARD == 'pseudo_c3':
    _c3 = pd.read_parquet(find('pseudo_c3.parquet'))
    _c3 = _c3[_c3.in_scope].reset_index(drop=True)
    _tr = pd.read_parquet(find('train.parquet'))
    _rows = [int(x) for v in _c3.train_rows for x in v]
    _mid_of = {int(x): m for m, v in zip(_c3.molecule_id, _c3.train_rows) for x in v}
    assert _rows and max(_rows) < len(_tr) and len(set(_rows)) == len(_rows), (len(_tr), len(_rows))
    _vq = _tr.iloc[_rows].copy()
    _vq['molecule_id'] = [_mid_of[_x] for _x in _rows]
    _tk = dict(zip(_c3.molecule_id, _c3.inchikey14))
    _okq = int((_vq.inchikey14.astype(str).values == _vq.molecule_id.map(_tk).values).sum())
    print(f'C3 query rows carrying the truth IK14 {_okq} of {len(_vq)}', flush=True)
    _tcols = list(pd.read_parquet(find('test.parquet')).columns)
    for _c in _tcols:
        if _c not in _vq.columns:
            _vq[_c] = None
    # ban keys = plain IK14 (cohort lists) U tautomer-canonical score_key (Engine 1 keys its pool by score_key)
    sys.path.insert(0, V4CODE)
    from casmi import chem as _hx_chem
    from rdkit import Chem as _Chem, RDLogger as _RDL
    from rdkit.Chem.Descriptors import ExactMolWt as _emw
    _RDL.DisableLog('rdApp.*')
    _plain = set(_c3.inchikey14) | {k for v in _c3.lib_ban_keys for k in v} | {k for v in _c3.pool_ban_keys for k in v}
    _sk = {_hx_chem.score_key(s) for s in _c3.smiles} - {None, ''}
    HX_C3_BAN = _plain | _sk
    print(f'C3 ban keys plain {len(_plain)} score_key {len(_sk)} union {len(HX_C3_BAN)}', flush=True)
    # library ban: rows by plain IK14, then tautomer rows (score_key) among rows at a truth's neutral mass (10 ppm)
    _hit = _tr.inchikey14.astype(str).isin(HX_C3_BAN).values
    _tm = np.sort(np.array([_emw(_Chem.MolFromSmiles(s)) for s in _c3.smiles], np.float64))
    _nm = _hx_chem.neutral_mass(_tr.precursor_mz.values.astype(np.float64), _tr.adduct.values)
    _j = np.clip(np.searchsorted(_tm, _nm), 1, len(_tm) - 1)
    _near = np.minimum(np.abs(_nm - _tm[_j - 1]), np.abs(_nm - _tm[_j])) <= 10e-6 * np.maximum(np.nan_to_num(_nm), 1.0)
    _smi = next(c for c in ('normalized_smiles', 'smiles') if c in _tr.columns)
    _cand = pd.unique(_tr[_smi].values[_near & ~_hit])
    _skm = {s: _hx_chem.score_key(s) for s in _cand if isinstance(s, str)}
    _taut = _tr[_smi].map(_skm).isin(HX_C3_BAN).values & _near & ~_hit
    _keep = ~(_hit | _taut)
    print(f'C3 library ban rows by plain key {int(_hit.sum())} + tautomer {int(_taut.sum())} '
          f'(score_key computed for {len(_skm)} structures near a truth mass)', flush=True)
    assert not _keep[_rows].any(), 'a query row survived the library ban'
    VD = '/kaggle/working/val_comp'; os.makedirs(VD, exist_ok=True)
    _vq[_tcols].to_parquet(os.path.join(VD, 'test.parquet'))
    _tr[_keep].reset_index(drop=True).to_parquet(os.path.join(VD, 'train.parquet'))
    _ids = list(_c3.molecule_id)
    pd.DataFrame({'molecule_id': _ids, 'smiles': 'CCO'}).to_csv(os.path.join(VD, 'sample_submission.csv'), index=False)
    json.dump(dict(pool_ban=sorted(HX_C3_BAN), masses=_tm.tolist()), open('/kaggle/working/hx_ban.json', 'w'))
    os.environ['HX_BAN_KEYS'] = '/kaggle/working/hx_ban.json'          # Engine 2 hook + PubChem runner read it
    del _tr
    COMP, IS_RERUN, ICE_BUDGET = VD, True, 5400
    print(f'VAL MODE pseudo_c3: molecules {len(_ids)} spectra {len(_vq)} banned train rows {int((~_keep).sum())}', flush=True)
'''

C3_POOL_CELL = r'''# === HARNESS (patch_harness_dump.py): pseudo_c3 Engine 1 pool ban (generated structures, pid -1, stay allowed) ===
import numpy as np, pandas as pd
if VAL_SHARD == 'pseudo_c3':
    _hx_bp = np.where(pd.Series(P.key).isin(HX_C3_BAN).values)[0]
    assert len(_hx_bp) > 0, 'HARNESS c3 Engine 1 pool ban matched no pool entry'
    _hx_w0 = P.window
    def _hx_window(target, ppm, _w0=_hx_w0, _bp=_hx_bp):
        w = _w0(target, ppm)
        return w[~np.isin(w, _bp)]
    P.window = _hx_window
    for _hx_k in HX_C3_BAN:
        P.key2pid.pop(_hx_k, None)
    print(f'HARNESS c3 ban engine1 pool pids {len(_hx_bp)}', flush=True)
'''

C3_CHECK_CELL = r'''# === HARNESS (patch_harness_dump.py): pseudo_c3 ban audit on the post-scoring lists (BASE, PC, ENG) ===
if VAL_SHARD == 'pseudo_c3':
    # a banned key in BASE is allowed only when HX_PID proves Engine 1 generated it (pid -1); unknown = pool = fail
    _hx_bk = {(m, k) for m, v in BASE.items() for k in v[1] if k in HX_C3_BAN}
    _hx_bg = {m for m, k in _hx_bk if HX_PID.get(m, {}).get(k, 0) < 0}
    _hx_bp = {m for m, k in _hx_bk if HX_PID.get(m, {}).get(k, 0) >= 0}
    _hx_p = {m for m, v in PC.items() for k in (v.get('pc_keys') or []) if k in HX_C3_BAN}
    _hx_e = {m for m, v in ENG.items() for k in (v.get('keys') or []) if k in HX_C3_BAN}
    print(f'HARNESS c3 audit: BASE lists holding a banned key: generated (pid -1, allowed) {len(_hx_bg)}, '
          f'pool {len(_hx_bp)}; PC lists {len(_hx_p)}, ENG lists {len(_hx_e)}', flush=True)
    assert not _hx_bp and not _hx_p and not _hx_e, \
        f'banned truth keys from a pool: BASE {sorted(_hx_bp)[:5]} PC {sorted(_hx_p)[:5]} ENG {sorted(_hx_e)[:5]}'
'''

# C7 cell literal (pseudo_c3 only): Engine 1 provenance per listed key, before the v19 re-rank reorders keys
C7_PID_OLD = "    if POOLPOP_MU > 0 and len(scs) > 1:"
C7_PID_NEW = ("    HX_PID[mid] = dict(zip(keys, pids_))          # HARNESS pseudo_c3: provenance for the ban audit\n"
              "    if POOLPOP_MU > 0 and len(scs) > 1:")

# PubChem cell literals (string patch, pseudo_c3 only): best_pool_fz without banned pool rows; probe never lists a ban key
PC_BEST_OLD = """            F = np.unpackbits(np.asarray(pfp[a:b]), axis=1)[:, :nbits].astype(np.float64)
            best = float((F @ z.astype(np.float64)).max())
"""
PC_BEST_NEW = """            F = np.unpackbits(np.asarray(pfp[a:b]), axis=1)[:, :nbits].astype(np.float64)
            if os.environ.get('HX_BAN_KEYS'):                       # HARNESS pseudo_c3: truth out of the pool window
                if 'HX_PK' not in globals():
                    import json as _hx_json
                    _hx_ban = np.array(sorted(_hx_json.load(open(os.environ['HX_BAN_KEYS']))['pool_ban']), dtype=object)
                    globals()['HX_PK'] = np.isin(pd.read_parquet(os.path.join(POOL_DIR, 'pool_meta.parquet'),
                                                                 columns=['key']).key.values, _hx_ban)
                F = F[~HX_PK[a:b]]
            best = float((F @ z.astype(np.float64)).max()) if len(F) else float('nan')
"""
PC_KEYS_OLD = """    _W['pool_keys'] = set(pd.read_parquet(pool_meta_path, columns=['key']).key.values.tolist())
"""
PC_KEYS_NEW = """    _W['pool_keys'] = set(pd.read_parquet(pool_meta_path, columns=['key']).key.values.tolist())
    if os.environ.get('HX_BAN_KEYS'):                               # HARNESS pseudo_c3: a ban key is never listed
        import json as _hx_json
        _W['pool_keys'] |= set(_hx_json.load(open(os.environ['HX_BAN_KEYS']))['pool_ban'])
"""


def esc(block):
    """Python source -> its escaped form inside a single-quoted notebook string literal (patch_v4n_mutator.esc)."""
    return block.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")


# Engine 2 runner literal (pseudo_c3 only): export the c3v2 mutator anchors, i.e. per molecule the top shifted spectral
# analogs as (pool smiles, exact mass, similarity) from the BANNED Engine 2 pool, plus the exact-window library max
# (patch_c3v2.py's export, unchanged). Read offline by build/novel_p4_inputs.py.
ENG_C3_OLD = """    json.dump(res, open(out, 'w'))
"""
ENG_C3_NEW = """    c3 = {}
    for r in recs:
        _a = r.get('ana') or []
        _l = r.get('lib')
        c3[str(r['mid'])] = dict(
            target=float(r.get('target') or 0.0),
            lib_exact=(float(np.max(_l)) if _l is not None and len(_l) else 0.0),
            ana=[[str(E.POOL['smiles'][int(_p)]), float(E.POOL['mass'][int(_p)]),
                  float(_s)] for _p, _s in _a[:20]])
    json.dump(c3, open(out + '.c3.json', 'w'))
    print('HARNESS c3 analog export', len(c3), sum(bool(v['ana']) for v in c3.values()), flush=True)
    json.dump(res, open(out, 'w'))
"""


def c3_graft(nb):
    """pseudo_c3 only: query cell after the VAL cell, Engine 1 pool ban after the bank cell, PubChem literal bans,
    and a ban audit just before the post-scoring snapshot."""
    i = find(nb, lambda s: s.startswith("# === HARNESS (patch_harness_dump.py): val mode."), "VAL cell")
    nb["cells"].insert(i + 1, code_cell(C3_VAL_CELL))
    b = find(nb, lambda s: s.startswith("# === HARNESS (patch_harness_dump.py): BASE bank"), "bank cell")
    nb["cells"].insert(b + 1, code_cell(C3_POOL_CELL))
    e2 = find(nb, lambda s: s.startswith("# Our engine (two-ranker"), "Engine 2 cell")
    set_src(nb["cells"][e2], must_replace(src_of(nb["cells"][e2]), esc(ENG_C3_OLD), esc(ENG_C3_NEW), "engine c3 export"))
    pc = find(nb, lambda s: "pc_runner.py" in s and "RUNNER" in s, "PubChem cell")
    t = must_replace(src_of(nb["cells"][pc]), esc(PC_BEST_OLD), esc(PC_BEST_NEW), "runner best_pool_fz ban")
    t = must_replace(t, esc(PC_KEYS_OLD), esc(PC_KEYS_NEW), "probe pool_keys ban")
    set_src(nb["cells"][pc], t)
    c7 = find(nb, lambda s: "BASE, n_err = {}, 0" in s, "C7 cell")
    set_src(nb["cells"][c7], must_replace(src_of(nb["cells"][c7]), C7_PID_OLD, C7_PID_NEW, "C7 pid provenance"))
    j = find(nb, lambda s: SNAP_ANCHOR in s, "simulator cell")
    nb["cells"].insert(j, code_cell(C3_CHECK_CELL))
    return nb


def build(shard: str, src: str = SRC, ranker=None) -> dict:
    if shard not in SHARDS:
        raise SystemExit(f"--shard must be one of {SHARDS}")
    if shard in NP_SHARDS and not (ranker and ranker[0] and ranker[1]):
        raise SystemExit(f"{shard}: pass the Engine 2 ranker query-set field and np label (--ranker-field/--ranker-drop), "
                         "as printed by the A0 kernel's NPZ inspection; never guess them")
    rfield, rdrop, rnames = (tuple(ranker) + ('',))[:3] if ranker else ('', '', '')
    nb = json.load(open(src, encoding="utf-8"))
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    i = find(nb, lambda s: s.startswith("# SMOKE switch"), "commit switch cell")
    nb["cells"].insert(i + 1, code_cell(VAL_CELL.replace("@SHARD@", repr(shard))))
    e2 = find(nb, lambda s: s.startswith("# Our engine (two-ranker"), "Engine 2 cell")
    nb["cells"].insert(e2 + 1, code_cell(RANKER_REPORT_CELL))
    nb["cells"].insert(e2, code_cell(RANKER_CELL.replace("@RFIELD@", repr(rfield)).replace("@RDROP@", repr(rdrop))
                                     .replace("@RNAMES@", repr(rnames))
                                     .replace("@HOOK@", repr(HX_HOOK_SRC))))
    b = find(nb, lambda s: "E = Engine(L, P, tfp" in s, "engine setup cell")
    nb["cells"].insert(b + 1, code_cell(BANK_CELL))
    g = find(nb, lambda s: "Engine 2 did not cover all active molecules" in s, "C12 final-pass cell")
    nb["cells"].insert(g, code_cell(ENG_GUARD_CELL))
    j = find(nb, lambda s: SNAP_ANCHOR in s, "simulator cell")
    set_src(nb["cells"][j], must_replace(src_of(nb["cells"][j]), SNAP_ANCHOR, SNAP + SNAP_ANCHOR, "C8 rerank loop"))
    nb = lib_sim_graft(nb)
    if shard == C3_SHARD:
        nb = c3_graft(nb)
    k = find(nb, lambda s: "def _claw_preseed_site" in s, "ICEBERG preseed cell")
    set_src(nb["cells"][k], must_replace(src_of(nb["cells"][k]), PRESEED_OLD, PRESEED_NEW, "cp313 rdkit preseed"))
    post_src = open(os.path.join(HERE, "casmi_post.py"), encoding="utf-8").read()
    nb["cells"].append(code_cell(f"POST_SRC = {post_src!r}\n" + FINAL_CELL))
    return nb


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="")
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ranker-field", default="")
    ap.add_argument("--ranker-drop", default="")
    ap.add_argument("--ranker-names", default="", help="name table the field codes index (e.g. sets)")
    a = ap.parse_args(argv)
    out = a.out or os.path.join(ROOT, f"casmi-0433-harness{('-' + a.shard.lower().replace('_', '')) if a.shard else ''}.ipynb")
    nb = build(a.shard, a.src, ranker=(a.ranker_field, a.ranker_drop, a.ranker_names))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
