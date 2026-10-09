"""Calibration notebook for the c3v2 gate (track B of specs/track4_priors_probes/c3v2_plan.md).

Built from the c3v2 submission notebook so it runs the *same* engine export, z, mutator and stage-1
gate code -- only the inputs change:

  * test.parquet  <- spectra of cohort molecules taken from train.parquet:
        pseudo_c3    (positives)      library rows AND pool entries of the truth banned
        c2_displace  (Class-2 negs)   library rows banned, structure kept in the pools
        Class-1 ctrl (random train)   nothing banned
  * train.parquet <- train.parquet minus every banned library row (written once, streamed)
  * engine A pool and the PubChem-runner pool window drop `pool_ban`; c2 structures removed with
    their library rows are put back into engine A's pool (`pool_add`), so Class 2 stays Class 2
  * the PubChem probe is skipped (fz_top = None), so `sure` here is an upper bound
  * the v4n base, ICEBERG/GLACIER and fusion cells are dropped: this calibrates stage 1 and the
    generator; stage 2 is measured on the LB counters

Output: /kaggle/working/c3cal_rows.csv (one row per molecule) and a printed verdict against the
pre-registered thresholds in specs/track4_priors_probes/gate_report.md.

Kaggle: attach every dataset of the 0.415 notebook plus a dataset holding p1_novel/pseudo_c3.parquet
and p1_novel/c2_displace.parquet.  Save Version (commit run) only -- never submit it.

usage: python build/patch_c3cal.py [--n-c3 300] [--n-c2 200] [--n-c1 100] [--out nb]
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import patch_c3v2 as P  # noqa: E402
from patch_v4n_mutator import code, esc, md, must_replace, set_src, src_of  # noqa: E402

DST = os.path.join(ROOT, "casmi26-c3cal.ipynb")

CAL_BUILD = '''# === C3 CALIBRATION: pseudo-test from train.parquet (replaces the commit switch) ===
import pandas as pd, numpy as np, hashlib, pyarrow as pa, pyarrow.parquet as pq
CAL_N_C3, CAL_N_C2, CAL_N_C1, CAL_MAX_SPEC = {N_C3}, {N_C2}, {N_C1}, 16
IS_RERUN = True                     # a calibration run is never submitted
CAL = '/kaggle/working/cal'; os.makedirs(CAL, exist_ok=True)
_c3 = pd.read_parquet(find('pseudo_c3.parquet')); _c2 = pd.read_parquet(find('c2_displace.parquet'))
if 'in_scope' in _c3:
    _c3 = _c3[_c3.in_scope]
_c3, _c2 = _c3.head(CAL_N_C3), _c2.head(CAL_N_C2)
_lib_ban = set(k for v in list(_c3.lib_ban_keys) + list(_c2.lib_ban_keys) for k in v)
_pool_ban = set(k for v in _c3.pool_ban_keys for k in v)
_want = {k: ('c3', m) for k, m in zip(_c3.inchikey14, _c3.molecule_id)}
_want.update({k: ('c2', m) for k, m in zip(_c2.inchikey14, _c2.molecule_id)})
_smiles = dict(zip(_c3.inchikey14, _c3.smiles)); _smiles.update(zip(_c2.inchikey14, _c2.smiles))

_src = pq.ParquetFile(os.path.join(COMP, 'train.parquet'))
_names = _src.schema_arrow.names
_ce = next((c for c in ('collision_energy_ev', 'collision_energy') if c in _names), None)
_qcols = ['inchikey14', 'normalized_smiles', 'adduct', 'precursor_mz', 'ionization_mode', 'ms2_mzs',
          'ms2_normalized_intensities', 'instrument_type'] + ([_ce] if _ce else [])
_writer, _q, _nspec, _c1 = None, [], {}, {}
for _g in range(_src.num_row_groups):
    _t = _src.read_row_group(_g)
    _ik = np.array(_t.column('inchikey14').cast(pa.string()).to_pylist(), dtype=object)
    _keep = ~pd.Series(_ik).isin(_lib_ban).values
    _tf = _t.filter(pa.array(_keep))
    if _writer is None:
        _writer = pq.ParquetWriter(os.path.join(CAL, 'train.parquet'), _tf.schema)
    _writer.write_table(_tf)
    _sub = _t.select(_qcols).to_pandas()
    for _i, _k in enumerate(_ik):
        if _k in _want:
            _coh, _m = _want[_k]
        elif (len(_c1) < CAL_N_C1 or _k in _c1) and int(hashlib.md5(str(_k).encode()).hexdigest(), 16) % 997 == 0:
            _c1.setdefault(_k, 'C1_%05d' % len(_c1)); _coh, _m = 'c1', _c1[_k]
            _smiles.setdefault(_k, _sub.normalized_smiles.iat[_i])
        else:
            continue
        if _nspec.get(_m, 0) >= (CAL_MAX_SPEC if _coh != 'c1' else 4):
            continue
        _nspec[_m] = _nspec.get(_m, 0) + 1
        _row = _sub.iloc[_i].to_dict(); _row.update(molecule_id=_m, cohort=_coh,
                                                    spectrum_id='%s_%d' % (_m, _nspec[_m]))
        _q.append(_row)
    del _t, _tf, _sub
_writer.close()
_te = pd.DataFrame(_q)
if _ce and _ce != 'collision_energy_ev':
    _te['collision_energy_ev'] = _te[_ce]
elif not _ce:
    _te['collision_energy_ev'] = None
_te.drop(columns=['inchikey14', 'normalized_smiles', 'cohort'] + ([_ce] if _ce and _ce != 'collision_energy_ev' else [])).to_parquet(os.path.join(CAL, 'test.parquet'))
pd.DataFrame({'molecule_id': sorted(_te.molecule_id.unique()), 'smiles': 'CCO'}).to_csv(
    os.path.join(CAL, 'sample_submission.csv'), index=False)
_tr = _te.drop_duplicates('molecule_id')[['molecule_id', 'cohort', 'inchikey14']].copy()
_tr['smiles'] = _tr.inchikey14.map(_smiles)
_tr.to_csv(os.path.join(CAL, 'truth.csv'), index=False)
# c2 structures lose their training spectra with the library ban; put them back into engine A's pool
_add = [[k, _smiles[k]] for k in _c2.inchikey14 if isinstance(_smiles.get(k), str)]
json.dump(dict(pool_ban=sorted(_pool_ban), pool_add=_add), open(os.path.join(CAL, 'ban.json'), 'w'))
os.environ['CASMI_BAN_KEYS'] = os.path.join(CAL, 'ban.json')
COMP = CAL
ICE_BUDGET = 0
print('C3CAL pseudo-test:', _tr.cohort.value_counts().to_dict(), '| spectra', len(_te),
      '| library rows banned for', len(_lib_ban), 'keys | pool ban', len(_pool_ban), '| c2 re-added', len(_add))
print('RERUN', IS_RERUN, '| COMP', COMP)
'''

ENG_BAN = '''    _ban_f = os.environ.get('CASMI_BAN_KEYS', '')
    if _ban_f and os.path.exists(_ban_f):        # C3 CALIBRATION: pool ban + c2 re-add
        import pandas as pd
        _bj = json.load(open(_ban_f)); _BAN = set(_bj.get('pool_ban', [])); _ADD = _bj.get('pool_add', [])
        _orig_bp = E.build_pool
        def _bp(L, workers):
            Lx = dict(L)
            if _ADD:
                Lx['ik'] = np.concatenate([L['ik'], np.asarray([a[0] for a in _ADD], dtype=object)])
                Lx['smi'] = np.concatenate([L['smi'], np.asarray([a[1] for a in _ADD], dtype=object)])
            P = _orig_bp(Lx, workers)
            keep = ~pd.Series(P['keys']).isin(_BAN).values
            P = dict(fp=P['fp'][keep], mass=P['mass'][keep], keys=P['keys'][keep], smiles=P['smiles'][keep])
            P['k2i'] = pd.Series(np.arange(len(P['keys'])), index=P['keys'])
            P['k2i'] = P['k2i'][~P['k2i'].index.duplicated()]
            print('C3CAL engine pool: removed', int((~keep).sum()), 'banned, re-added', len(_ADD), flush=True)
            return P
        E.build_pool = _bp
'''

PC_POOL_LOAD = '''    _ban_f = os.environ.get('CASMI_BAN_KEYS', '')
    _BANL = (np.array(sorted(json.load(open(_ban_f)).get('pool_ban', [])), dtype=object)
             if _ban_f and os.path.exists(_ban_f) else np.array([], dtype=object))
    pkeys = (pd.read_parquet(os.path.join(POOL_DIR, 'pool_meta.parquet'), columns=['key']).key.values
             if len(_BANL) else None)
    print('C3CAL pool-window ban:', len(_BANL), 'keys', flush=True)
'''

PC_F_OLD = '''            F = np.unpackbits(np.asarray(pfp[a:b]), axis=1)[:, :nbits].astype(np.float64)
            best = float((F @ z.astype(np.float64)).max())
'''
PC_F_NEW = '''            F = np.unpackbits(np.asarray(pfp[a:b]), axis=1)[:, :nbits].astype(np.float64)
            if pkeys is not None:
                F = F[~np.isin(pkeys[a:b], _BANL)]
            best = float((F @ z.astype(np.float64)).max()) if len(F) else float('nan')
'''

EVAL_CELL = '''# === C3 CALIBRATION: per-molecule rows and the pre-registered verdict ===
import collections
truth = pd.read_csv('/kaggle/working/cal/truth.csv')
MUTF = json.load(open('/kaggle/working/mut_lists.json')) if os.path.exists('/kaggle/working/mut_lists.json') else {}
DIAG = {str(d['molecule_id']): d for d in json.load(open('/kaggle/working/mut_lists.json.diag.json'))} \\
    if os.path.exists('/kaggle/working/mut_lists.json.diag.json') else {}
C3D = json.load(open('/kaggle/working/eng_lists.json.c3.json'))
sys.path.insert(0, V3CODE)
from casmi import chem as _chem
_bits = np.load(os.path.join(POOL_DIR, 'fp_bits.npy'))
_zf = np.load('/kaggle/working/pc_z.npz', allow_pickle=True)
_Z = {str(m): _zf['z'][i] for i, m in enumerate(_zf['mids'])}
def _fz(s, z):
    fp = _chem.raw_fingerprint(s) if isinstance(s, str) else None
    return None if fp is None or z is None else float(fp[_bits].astype(np.float32) @ z)
rows = []
for r in truth.itertuples():
    m = str(r.molecule_id); d = DIAG.get(m, {}); u = MUTF.get(m, {}); c = C3D.get(m, {})
    gk = [str(k)[:14] for k in u.get('keys', [])]
    ek = [str(k)[:14] for k in (ENG.get(m) or {}).get('keys', [])]
    z = _Z.get(m)
    a0 = (c.get('ana') or [[None]])[0][0]
    rows.append(dict(molecule_id=m, cohort=r.cohort, tier=d.get('tier') or '', gate=d.get('gate', ''),
                     n_anchors=d.get('n_anchors', 0), delta_steps=d.get('delta_steps', -1),
                     ana_max=d.get('ana_max'), lib_exact=d.get('lib_exact'), n_gen=d.get('n_gen', 0),
                     gen_rank=(gk.index(r.inchikey14) + 1 if r.inchikey14 in gk else 0),
                     eng_rank=(ek.index(r.inchikey14) + 1 if r.inchikey14 in ek else 0),
                     fz_truth=_fz(r.smiles, z), fz_anchor0=_fz(a0, z), fz_best=d.get('fz_best'),
                     best_pool_fz=d.get('best_pool_fz'), fz_sep=d.get('fz_sep')))
D = pd.DataFrame(rows)
D.to_csv('/kaggle/working/c3cal_rows.csv', index=False)
print('C3CAL rows ->', '/kaggle/working/c3cal_rows.csv', len(D))
V = {}
for coh, g in D.groupby('cohort'):
    fired = g.tier.isin(['sure', 'likely'])
    n = len(g)
    V[coh] = dict(n=n, fire=fired.mean(), sure=(g.tier == 'sure').mean(),
                  gen_any=(g.gen_rank > 0).mean(), gen_top3=((g.gen_rank > 0) & (g.gen_rank <= 3)).mean(),
                  q3_fired=(((g.gen_rank > 0) & (g.gen_rank <= 3)) & fired).sum() / max(1, fired.sum()),
                  eng_top1=(g.eng_rank == 1).mean(),
                  displaced=(fired & (g.eng_rank == 1)).mean(),
                  i0=(g.fz_truth > g.fz_anchor0).mean() if g.fz_anchor0.notna().any() else float('nan'),
                  anchors=(g.n_anchors > 0).mean())
    print(f'{coh}: ' + ' '.join(f'{k}={v:.3f}' if isinstance(v, float) else f'{k}={v}' for k, v in V[coh].items()))
    print('   gate funnel:', g[~fired].gate.value_counts().head(8).to_dict())
print()
print('PRE-REGISTERED (gate_report.md, c3v2):')
_c1, _c2, _c3 = V.get('c1', {}), V.get('c2', {}), V.get('c3', {})
_chk = [('C1 fire rate <= 1%', _c1.get('fire', 1) <= 0.01), ('C1 sure == 0', _c1.get('sure', 1) == 0),
        ('C2 fire rate <= 3%', _c2.get('fire', 1) <= 0.03),
        ('C2 displaced top-1 <= 0.5%', _c2.get('displaced', 1) <= 0.005),
        ('C3 q3|fired > 1.25 x C2 fire rate', _c3.get('q3_fired', 0) > 1.25 * _c2.get('fire', 1)),
        ('I0: fz(truth) > fz(anchor) on >= 70% of C3', _c3.get('i0', 0) >= 0.70)]
for _n, _ok in _chk:
    print(('  PASS ' if _ok else '  FAIL ') + _n)
'''


def main(argv):
    opts = {"n_c3": "300", "n_c2": "200", "n_c1": "100", "out": DST}
    i = 0
    while i < len(argv):
        k = argv[i].lstrip("-").replace("-", "_")
        if k not in opts:
            raise SystemExit("unknown flag " + argv[i])
        opts[k] = argv[i + 1]
        i += 2
    tmp = os.path.join(tempfile.mkdtemp(), "c3v2.ipynb")
    P.main(["--out", tmp])
    nb = json.load(open(tmp, encoding="utf-8"))

    def find(pred, what):
        hits = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code" and pred(src_of(c))]
        if len(hits) != 1:
            raise SystemExit(f"PATCH FAILED: {what} matched {len(hits)} cells")
        return hits[0]

    i_smoke = find(lambda s: "SMOKE switch" in s, "commit-switch cell")
    i_eng = find(lambda s: "eng_runner.py" in s, "engine cell")
    i_pc = find(lambda s: "pc_runner.py" in s and "RUNNER" in s, "PubChem cell")
    i_mut = find(lambda s: P.MARK in s, "mutator cell")

    set_src(nb["cells"][i_smoke], CAL_BUILD.replace("{N_C3}", opts["n_c3"])
            .replace("{N_C2}", opts["n_c2"]).replace("{N_C1}", opts["n_c1"]))

    t = src_of(nb["cells"][i_eng])
    t = must_replace(t, esc("    E.find = _find\n"), esc("    E.find = _find\n" + ENG_BAN), "engine pool ban")
    set_src(nb["cells"][i_eng], t)

    t = src_of(nb["cells"][i_pc])
    anchor = esc("    pfp = np.load(os.path.join(POOL_DIR, 'pool_fp.npy'), mmap_mode='r')\n")
    t = must_replace(t, anchor, anchor + esc(PC_POOL_LOAD), "pool ban load")
    t = must_replace(t, esc(PC_F_OLD), esc(PC_F_NEW), "pool window mask")
    a = t.find(esc("    res = {}\n    import multiprocessing as mp\n"))
    b = t.find(esc("    np.savez_compressed(os.path.join(os.path.dirname(OUT), 'pc_z.npz'),\n"))
    if a < 0 or b < 0 or b < a:
        raise SystemExit("PATCH FAILED: PubChem probe block not found")
    tail = ("    res = {mid: dict(pc=[], pc_fz=[], pc_keys=[], fz_top=None, **extra[mid]) for mid, _t, _z in tasks}\n"
            "    print('C3CAL: PubChem probe skipped (fz_top=None -> sure is an upper bound)', flush=True)\n")
    t = t[:a] + esc(tail) + t[b:]
    set_src(nb["cells"][i_pc], t)

    t = src_of(nb["cells"][i_mut])
    t = must_replace(t, "_env = dict(os.environ, MUT_V2='1',", "_env = dict(os.environ, MUT_V2='1', MUT_KEEP_DECLINED='1',",
                     "keep declined")
    t = must_replace(t, "if MUT_ON and PC and os.path.exists(", "if MUT_ON and os.path.exists(", "PC guard")
    set_src(nb["cells"][i_mut], t)

    nb["cells"] = nb["cells"][:i_mut + 1] + [code(EVAL_CELL)]
    nb["cells"].insert(0, md(
        "# C3 gate calibration (c3v2, track B) -- commit run only, never submit\n"
        "Cohort spectra from `train.parquet` stand in for the test set; cohort truths are banned from "
        "the library (and, for pseudo_c3, from the pools). Same engine export, z, mutator and stage-1 "
        "gate code as `casmi26-c3v2-*.ipynb`. Output: `c3cal_rows.csv` and a PASS/FAIL verdict against "
        "the pre-registered thresholds."))
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    json.dump(nb, open(opts["out"], "w", encoding="utf-8"), indent=1)
    print("wrote %s (%d cells; c3=%s c2=%s c1=%s)" % (opts["out"], len(nb["cells"]), opts["n_c3"],
                                                     opts["n_c2"], opts["n_c1"]))
    return opts["out"]


if __name__ == "__main__":
    main(sys.argv[1:])
