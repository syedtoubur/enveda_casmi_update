"""Tests for build/fold_audit.py. Run: python build/test_fold_audit.py

Synthetic inputs only (no real checkpoints): item building, the +-10 ppm window, truth insertion, ranking, the
pre-registered verdict, and a dry run of main() end to end with a stub ModelBank whose fpnet_0 is an oracle on
one library and an anti-oracle on another -- the audit must call the first `yes` and the second `no`.
"""
import os
import sys
import tempfile
import warnings

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CODE = os.environ.get('CASMI_V4CODE', 'E:/ENVEDA_casmi/.scratch/casmi_v4')
os.environ.setdefault('NUMBA_CACHE_DIR', os.path.join(tempfile.gettempdir(), 'fold_audit_numba'))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import fold_audit as FA

chem, fpnet, poolmod, spectra = FA.import_casmi(CODE)
FAIL = []
PVFP = os.environ.get('PVFP_SRC', '')   # path to the base notebook's Engine 2 pv_fp.py (optional)


def check(cond, msg):
    print(f'  {"ok  " if cond else "FAIL"} {msg}')
    if not cond:
        FAIL.append(msg)


PROTON = chem.PROTON
BITS = np.arange(0, chem.FP_RAW_BITS, 7).astype(np.int32)      # informative-bit selection stand-in
# isomer groups -> multi-candidate windows
POOL_SMILES = ['CCO', 'COC',                                       # C2H6O
               'CCCO', 'CC(C)O', 'CCOC',                           # C3H8O
               'CCCCO', 'CC(C)CO', 'CCOCC', 'CC(O)CC',             # C4H10O
               'c1ccccc1O', 'Cc1ccccc1', 'CC(=O)O', 'OCC(O)CO']
MISSING = 'CCCOC'                                                  # C4H10O, not in the pool -> must be added


def fp_of(smi):
    return chem.raw_fingerprint(smi)[BITS].astype(np.uint8)


def mass_of(smi):
    return chem.mol_props(smi)[1]


def ik(i):
    return f'IKFAKE{i:08d}'


def make_pool(d):
    recs = [dict(smiles=s, mass=mass_of(s), formula=chem.mol_props(s)[0], key=ik(i)) for i, s in enumerate(POOL_SMILES)]
    meta = pd.DataFrame(recs).sort_values('mass', kind='mergesort').reset_index(drop=True)
    meta['n_heavy'] = 3; meta['src'] = 0
    meta['train_sid'] = [int(k[-8:]) for k in meta.key]           # sid == original index
    fps = np.stack([np.packbits(fp_of(s)) for s in meta.smiles])
    meta.to_parquet(f'{d}/pool_meta.parquet', index=False)
    np.save(f'{d}/pool_fp.npy', fps); np.save(f'{d}/fp_bits.npy', BITS)
    ts = pd.DataFrame(dict(inchikey14=[ik(i) for i in range(len(POOL_SMILES))] + ['IKMISSING00000'],
                           sid=list(range(len(POOL_SMILES))) + [99]))
    ts.to_parquet(f'{d}/train_structs.parquet', index=False)
    return meta


def spec_row(lib, key, smi, adduct='[M+H]+', ce=(20.0, 40.0)):
    prec = mass_of(smi) + PROTON
    return dict(ingest_lib=lib, inchikey14=key, normalized_smiles=smi, adduct=adduct,
                ionization_mode='positive', precursor_mz=prec, collision_energy_ev=list(ce),
                ms2_mzs=[30.0, prec - 18.0, prec + 5.0], ms2_normalized_intensities=[0.5, 1.0, 0.9])


# ------------------------------------------------------------------------------------------------
def test_build_item():
    r = spec_row('gnps', 'X', 'CCCO')
    it = FA.build_item(r, chem, fpnet, spectra)
    check(len(it['mz']) == 2 and it['mz'].max() < r['precursor_mz'], 'peaks above prec+2 dropped')
    check(abs(it['it'].max() - 1.0) < 1e-6, 'sqrt intensity normalised to 1')
    check(it['ce'] == 30.0 and it['ce_known'] == 1.0 and it['n_merged'] == 2, 'ce mean / ce_known / n_merged=ce_n')
    check(it['mode'] == 1.0 and it['adduct_ix'] == chem.adduct_index('[M+H]+'), 'mode and adduct index')
    r2 = dict(r, collision_energy_ev=None, ionization_mode='negative', adduct='[M-H]-')
    it2 = FA.build_item(r2, chem, fpnet, spectra)
    check(it2['ce'] == 0.0 and it2['ce_known'] == 0.0 and it2['n_merged'] == 1 and it2['mode'] == -1.0,
          'unknown CE -> ce 0, ce_known 0, n_merged 1; negative mode')
    ref = fpnet.prep_peaks(np.array(r['ms2_mzs']), np.array(r['ms2_normalized_intensities']), r['precursor_mz'])
    check(np.allclose(ref[0], it['mz']) and np.allclose(ref[1], it['it']), 'matches prep_peaks on clean input')
    args = fpnet.collate([it, it2], 'cpu')
    check(args[0].shape[0] == 2 and len(args) == 9, 'items collate')


def test_sample():
    meta = pd.DataFrame(dict(
        row=np.arange(9),
        ingest_lib=['gnps', 'gnps', 'gnps', 'riken', 'riken', 'enveda-np-examples', 'gnps', 'riken', 'gnps'],
        inchikey14=['A', 'A', 'B', 'C', 'B', 'A', 'D', 'E', None],
        adduct=['[M+Na]+', '[M+H]+', '[M+H]+', '[M+H]+', '[M+H]+', '[M+H]+', '[M-H]-', '[M+H]+', '[M+H]+']))
    s = FA.sample_structures(meta, n=150, seed=0)
    g = s[s.lib == 'gnps']
    # A is in gnps and np-examples -> not gnps-unique; B in gnps and riken -> not unique; D unique
    check(sorted(g.inchikey14) == ['D'], 'gnps keeps only library-unique structures')
    check(sorted(s[s.lib == 'riken'].inchikey14) == ['C', 'E'], 'riken unique = C, E')
    npx = s[s.lib == 'enveda-np-examples']
    check(list(npx.inchikey14) == ['A'] and npx.shared.all() and not g.shared.any(),
          'np-examples has no unique structures -> sampled anyway, flagged shared')
    check(s.inchikey14.notna().all(), 'null ik14 rows ignored')
    meta2 = pd.DataFrame(dict(row=np.arange(4), ingest_lib=['msdial'] * 4, inchikey14=['Q', 'Q', 'Q', 'R'],
                              adduct=['[M+Na]+', '[M+H]+', '[M+H]+', '[M-H]-']))
    s2 = FA.sample_structures(meta2, n=150, seed=0)
    check(s2.set_index('inchikey14').loc['Q', 'row'] == 1, 'prefers [M+H]+, then lowest row')
    check(s2.set_index('inchikey14').loc['R', 'row'] == 3, 'falls back to other adducts')
    big = pd.DataFrame(dict(row=np.arange(500), ingest_lib=['mona'] * 500,
                            inchikey14=[f'K{i}' for i in range(500)], adduct=['[M+H]+'] * 500))
    a = FA.sample_structures(big, n=150, seed=0); b = FA.sample_structures(big, n=150, seed=0)
    check(len(a) == 150 and list(a.row) == list(b.row), 'caps at n; deterministic under seed 0')


def test_read_rows(d):
    rows = [spec_row('gnps', ik(i), POOL_SMILES[i % 5]) for i in range(10)]
    t = pa.Table.from_pylist(rows)
    p = f'{d}/rg.parquet'
    pq.write_table(t, p, row_group_size=3)
    check(pq.ParquetFile(p).metadata.num_row_groups == 4, 'fixture has 4 row groups')
    out = FA.read_rows(p, [8, 1, 4, 5])
    check(list(out.row) == [1, 4, 5, 8] and list(out.inchikey14) == [ik(1), ik(4), ik(5), ik(8)],
          'reads exactly the requested rows across row groups')
    check(abs(out.precursor_mz.iloc[3] - rows[8]['precursor_mz']) < 1e-9, 'row content intact')
    m = FA.read_meta(p)
    check(list(m.columns) == ['row'] + FA.META_COLS and len(m) == 10, 'meta has row + meta columns')


def test_window_rank(d):
    meta = make_pool(d)
    P = poolmod.Pool(d, load_frag=False, verbose=False)
    ts = pd.read_parquet(f'{d}/train_structs.parquet')
    ik2pid = FA.truth_maps(P, ts)
    check(len(ik2pid) == len(POOL_SMILES), 'train ik14 -> pid via sid')
    pid, efp, m, how = FA.resolve_truth(ik(2), 'CCCO', P, ik2pid, chem, True)
    check(how == 'pool' and P.smiles[pid] == 'CCCO' and abs(m - mass_of('CCCO')) < 1e-9, 'pool truth resolved')
    w = FA.window_ids(P, m, 10.0)
    check(sorted(P.smiles[w]) == sorted(['CCCO', 'CC(C)O', 'CCOC']), '+-10 ppm window = the C3H8O isomers')
    check(len(FA.window_ids(P, m * (1 + 11e-6), 10.0)) == 0, '11 ppm off -> empty window')
    check(len(FA.window_ids(P, m * (1 + 9e-6), 10.0)) == 3, '9 ppm off -> still inside')
    F, t, keys = FA.candidate_set(P, w, pid, None)
    check(len(F) == 3 and keys[t] == ik(2) and np.array_equal(F[t], fp_of('CCCO')), 'truth row located in window')
    F2, t2, _ = FA.candidate_set(P, np.array([i for i in w if i != pid]), pid, None)
    check(len(F2) == 3 and t2 == 2 and np.array_equal(F2[t2], fp_of('CCCO')), 'pool truth outside window is appended')
    pid3, efp3, m3, how3 = FA.resolve_truth('IKMISSING00000', MISSING, P, ik2pid, chem, True)
    check(how3 == 'added' and pid3 == -1 and np.array_equal(efp3, fp_of(MISSING)), 'missing truth -> rdkit fp on pool bits')
    F3, t3, k3 = FA.candidate_set(P, FA.window_ids(P, m3, 10.0), pid3, efp3)
    check(len(F3) == 5 and t3 == 4 and k3[t3] is None, 'missing truth appended to its 4-isomer window')
    check(FA.resolve_truth('NOPE', MISSING, P, ik2pid, chem, False)[3] == 'unresolved', 'no rdkit -> unresolved')
    # ranking
    check(FA.rank_of_truth([0.1, 0.9, 0.5], 1) == 1, 'best score -> rank 1')
    check(FA.rank_of_truth([0.9, 0.1, 0.5], 1) == 3, 'worst of 3 -> rank 3')
    check(FA.rank_of_truth([0.5, 0.5, 0.5], 1) == 1, 'ties do not count against the truth (window_mrr)')
    check(FA.rank_of_truth([0.9, 0.1, 0.5], 1, ['k', 'k', 'j']) == 2, 'duplicate of the truth key ignored')
    check(FA.rr_at(1) == 1.0 and FA.rr_at(25) == 0.04 and FA.rr_at(26) == 0.0, 'rr@25')
    z = (fp_of('CCCO').astype(np.float32) * 2 - 1)
    check(FA.rank_of_truth(F @ z, t, keys) == 1, 'oracle logits rank truth first (f.z)')
    check(FA.rank_of_truth(F @ -z, t, keys) == 3, 'anti-oracle logits rank truth last')


def test_verdict():
    check(FA.verdict(0.8, 1.0) == 'yes' and FA.verdict(0.79, 1.0) == 'unclear', 'yes boundary at 0.8x')
    check(FA.verdict(0.5, 1.0) == 'no' and FA.verdict(0.51, 1.0) == 'unclear', 'no boundary at 0.5x')
    check(FA.verdict(0.3, 0.0) == 'unclear' and FA.verdict(np.nan, 0.5) == 'unclear', 'degenerate -> unclear')


def test_dry_run(d):
    make_pool(d)
    # spectraverse: fpnet_0 oracle (trained); gnps: fpnet_0 anti-oracle (held out); np-examples shares a structure
    rows = [spec_row('spectraverse', ik(2), 'CCCO'), spec_row('spectraverse', ik(5), 'CCCCO'),
            spec_row('spectraverse', ik(0), 'CCO', adduct='[M+Na]+'),
            spec_row('gnps', ik(3), 'CC(C)O'), spec_row('gnps', ik(7), 'CCOCC'),
            spec_row('gnps', 'IKMISSING00000', MISSING),
            spec_row('enveda-np-examples', ik(2), 'CCCO'), spec_row('gnps', ik(1), 'COC')]
    rows[2]['precursor_mz'] = mass_of('CCO') + chem.ADDUCTS['[M+Na]+'][2]
    for i, r in enumerate(rows):                     # isomers share m/z: tag each spectrum by a unique CE
        r['collision_energy_ev'] = [10.0 + i]
    pq.write_table(pa.Table.from_pylist(rows), f'{d}/train.parquet', row_group_size=3)
    truth = {r['collision_energy_ev'][0]: fp_of(r['normalized_smiles']).astype(np.float32) * 2 - 1 for r in rows}
    lib_of = {r['collision_energy_ev'][0]: r['ingest_lib'] for r in rows}
    for n in ('fpnet_0.pt', 'fpnet_full1.pt'):
        open(f'{d}/{n}', 'w').close()

    class StubBank:
        def __init__(self, paths, device='cpu'):
            assert device == 'cpu'
            self.oracle_all = 'full1' in paths[0]; self.nbits = len(BITS)

        def logits(self, items, bs=64):
            out = []
            for x in items:
                k = x['ce']
                s = 1.0 if self.oracle_all or lib_of[k] != 'gnps' else -1.0
                out.append(s * truth[k])
            return np.stack(out)

    real = fpnet.ModelBank
    fpnet.ModelBank = StubBank
    try:
        summ = FA.main(['--train', f'{d}/train.parquet', '--code', CODE, '--fpnet0', f'{d}/fpnet_0.pt',
                        '--full1', f'{d}/fpnet_full1.pt', '--pool', d, '--out', f'{d}/out'])
    finally:
        fpnet.ModelBank = real
    s = summ.set_index('lib')
    check(set(s.index) == {'spectraverse', 'gnps', 'enveda-np-examples'}, 'one summary row per library sampled')
    check(s.loc['spectraverse', 'n'] == 2, 'CCCO is shared with np-examples -> not sampled for spectraverse')
    check(s.loc['spectraverse', 'trained_on'] == 'yes', 'oracle library -> trained_on=yes')
    check(s.loc['gnps', 'trained_on'] == 'no', 'anti-oracle library -> trained_on=no')
    check(s.loc['gnps', 'n'] == 4 and s.loc['gnps', 'n_added'] == 1, 'missing truth added and counted')
    check(bool(s.loc['enveda-np-examples', 'shared']) and not bool(s.loc['gnps', 'shared']), 'shared flag carried')
    check(s.loc['spectraverse', 'mrr_full1'] == 1.0 and s.loc['spectraverse', 'top1_fpnet_0'] == 1.0, 'oracle MRR 1')
    check(s.loc['gnps', 'med_cand'] >= 2, 'median candidate count reported')
    q = pd.read_csv(f'{d}/out/fold_audit_queries.csv')
    check(os.path.exists(f'{d}/out/fold_audit.csv') and len(q) == len(rows) - 1, 'csv outputs written')
    check((q.prec_ppm.abs() < 1).all(), 'precursor-derived neutral mass within 1 ppm of truth (Na adduct too)')


# ------------------------------------------------------------------------------------------------
# A0: np-examples structure- vs spectrum-level holdout (Option A, gate_report "Option A0")
# ------------------------------------------------------------------------------------------------
def np_meta():
    # np structures A, B (A also in gnps x3 + riken x1, B in mona x1); controls C..G library-unique elsewhere
    rows = [('enveda-np-examples', 'A', '[M+H]+'), ('enveda-np-examples', 'A', '[M+Na]+'),
            ('enveda-np-examples', 'B', '[M+H]+'),
            ('gnps', 'A', '[M+Na]+'), ('gnps', 'A', '[M+H]+'), ('gnps', 'A', '[M+H]+'), ('riken', 'A', '[M-H]-'),
            ('mona', 'B', '[M+H]+'),
            ('gnps', 'C', '[M+H]+'), ('gnps', 'D', '[M+H]+'), ('riken', 'E', '[M+H]+'), ('mona', 'F', '[M+H]+'),
            ('enveda-180', 'G', '[M+H]+'), ('gnps', 'H', '[M+H]+'), ('riken', 'H', '[M+H]+')]
    return pd.DataFrame(dict(row=np.arange(len(rows)), ingest_lib=[r[0] for r in rows],
                             inchikey14=[r[1] for r in rows], adduct=[r[2] for r in rows]))


def test_sample_np_sets():
    s = FA.sample_np_sets(np_meta(), k_other=2, n_control=4, seed=0)
    npq, oth, ctl = (s[s.qset == q] for q in ('np', 'other', 'control'))
    check(sorted(npq.row) == [0, 1, 2], 'np = every enveda-np-examples spectrum')
    check(set(oth.inchikey14) == {'A', 'B'} and (oth.lib != 'enveda-np-examples').all(),
          'other = np structures, other libraries only')
    a = oth[oth.inchikey14 == 'A']
    check(len(a) == 2 and set(a.lib) == {'gnps', 'riken'}, 'other: <= k per structure, distinct libraries first')
    check(int(a[a.lib == 'gnps'].row.iloc[0]) == 4, 'other: within a library prefer [M+H]+ then lowest row')
    check(len(oth[oth.inchikey14 == 'B']) == 1, 'other: fewer than k when the structure has fewer')
    check(len(ctl) == 4 and not set(ctl.inchikey14) & {'A', 'B', 'H'}, 'control: n structures, not np, library-unique')
    check(set(ctl.lib) <= {'gnps', 'riken', 'mona'}, 'control: drawn from the libraries of the other set')
    check(ctl.inchikey14.is_unique, 'control: one spectrum per structure')
    s2 = FA.sample_np_sets(np_meta(), k_other=2, n_control=4, seed=0)
    check(list(s.row) == list(s2.row), 'deterministic under seed')


def test_a0_decision():
    check(FA.a0_decision(0.8) == 'structures' and FA.a0_decision(0.5) == 'structures', '<= 0.8 -> structures held out')
    check(FA.a0_decision(0.9) == 'spectra' and FA.a0_decision(1.05) == 'spectra', '>= 0.9 -> spectra only')
    check(FA.a0_decision(0.85) == 'stop' and FA.a0_decision(np.nan) == 'stop', 'between / nan -> stop (human)')


def test_summarize_sets():
    df = pd.DataFrame(dict(qset=['np', 'np', 'other', 'other', 'control', 'control'],
                           truth=['pool'] * 6,
                           rank_full1=[1, 1, 1, 2, 1, 1], rank_fpnet_0=[2, 30, 2, 4, 1, 1],
                           rank_fe_A=[1, 1, 1, 2, 1, 1]))
    out = FA.summarize_sets(df, ['fpnet_0', 'full1', 'fe_A'], ref='full1').set_index(['qset', 'model'])
    check(abs(out.loc[('np', 'fpnet_0'), 'mrr'] - 0.25) < 1e-9 and out.loc[('np', 'fpnet_0'), 'top1'] == 0,
          'mrr@25 / top1 per set and model')
    check(abs(out.loc[('other', 'fpnet_0'), 'ratio'] - (0.375 / 0.75)) < 1e-9, 'ratio vs the reference model')
    check(bool(out.loc[('np', 'fe_A'), 'leak']) and not bool(out.loc[('np', 'fpnet_0'), 'leak']),
          'leak flag = ratio >= 0.9 on np')
    check(out.loc[('control', 'full1'), 'ratio'] == 1.0, 'reference ratio 1')


def test_pv_fingerprint():
    bits = np.array([0, 5, 4096, 8192 + 3, 8192 + 2048 + 10], np.int64)
    fp = FA.pv_fingerprint('CCCO', bits)
    check(fp is not None and fp.shape == (5,) and fp.dtype == np.uint8, 'pv fp on the given bit selection')
    check(FA.pv_fingerprint('not a smiles', bits) is None, 'bad smiles -> None')


def test_inspect_npz(d):
    p = f'{d}/r.npz'
    np.savez(p, X=np.zeros((5, 3), np.float32), Y=np.array([1, 0, 0, 1, 0]), S=np.array([0, 1, 0, 1, 1]),
             Q=np.array(['np', 'gnps', 'np', 'x', 'x']), blocks=np.array(['a', 'b']))
    info = FA.inspect_npz(p)
    check(info['X']['shape'] == (5, 3) and 'counts' not in info['X'], 'big 2-D arrays: header only')
    check(info['Q']['counts'] == {'np': 2, 'gnps': 1, 'x': 2}, 'small-cardinality 1-D arrays: value counts')
    check(info['S']['counts'] == {0: 2, 1: 3}, 'int arrays counted too')
    check(info['blocks']['values'] == ['a', 'b'] and 'values' not in info['X'], 'short arrays listed in order')


def test_a0_dry_run(d):
    make_pool(d)
    # np structure CCCO (ik 2) also in gnps; np structure CCCCO (ik 5) also in riken; controls elsewhere
    rows = [spec_row('enveda-np-examples', ik(2), 'CCCO'), spec_row('enveda-np-examples', ik(5), 'CCCCO'),
            spec_row('gnps', ik(2), 'CCCO'), spec_row('riken', ik(5), 'CCCCO'),
            spec_row('gnps', ik(3), 'CC(C)O'), spec_row('riken', ik(7), 'CCOCC'), spec_row('gnps', ik(1), 'COC')]
    for i, r in enumerate(rows):
        r['collision_energy_ev'] = [10.0 + i]; r['instrument_type'] = 'timsTOF'
    pq.write_table(pa.Table.from_pylist(rows), f'{d}/train.parquet', row_group_size=3)
    truth = {r['collision_energy_ev'][0]: fp_of(r['normalized_smiles']).astype(np.float32) * 2 - 1 for r in rows}
    is_np = {r['collision_energy_ev'][0]: r['inchikey14'] in (ik(2), ik(5)) for r in rows}

    def fake_logits(spec, name, path, family, code):
        if family == 'pv':
            return None, 'not scorable: stub'
        out = []
        for c in spec.collision_energy_ev:
            k = float(c[0])
            s = -1.0 if (name == 'fpnet_0' and is_np[k]) else 1.0      # fpnet_0 never saw the np structures
            out.append(s * truth[k])
        return np.stack(out), None

    models = [('full1', 'f1', 'fpnet', CODE), ('fpnet_0', 'f0', 'fpnet', CODE), ('fe_A', 'fa', 'fpnet6', CODE),
              ('fp_merged_m1', 'pm', 'pv', None)]
    summ, dec = FA.run_a0(f'{d}/train.parquet', d, models, fake_logits, k_other=2, n_control=3,
                          out=f'{d}/a0', pv_bits=None)
    s = summ.set_index(['qset', 'model'])
    check(dec == 'structures', 'anti-oracle fpnet_0 on np structures -> structures held out')
    check(s.loc[('other', 'fpnet_0'), 'ratio'] < 0.8 and s.loc[('control', 'fpnet_0'), 'ratio'] == 1.0,
          'other-library ratio low, controls 1.0')
    check(bool(s.loc[('np', 'fe_A'), 'leak']), 'fe_A as good as full1 on np -> declared residual leak')
    check(('np', 'fp_merged_m1') not in s.index, 'unscorable model reported, not forced')
    check(os.path.exists(f'{d}/a0/a0_summary.csv') and os.path.exists(f'{d}/a0/a0_queries.csv'), 'a0 csvs written')


def test_a0_pv_path(d):
    # a P-family model on its own bit space: candidates re-fingerprinted with pv_fingerprint (oracle -> rank 1)
    make_pool(d)
    rows = [spec_row('enveda-np-examples', ik(2), 'CCCO'), spec_row('gnps', ik(2), 'CCCO'),
            spec_row('gnps', ik(3), 'CC(C)O'), spec_row('enveda-np-examples', 'IKMISSING00000', MISSING),
            spec_row('riken', 'IKMISSING00000', MISSING)]
    for i, r in enumerate(rows):
        r['collision_energy_ev'] = [10.0 + i]; r['instrument_type'] = 'timsTOF'
    pq.write_table(pa.Table.from_pylist(rows), f'{d}/train.parquet')
    bits = np.arange(0, 10407, 3)
    smi = {r['collision_energy_ev'][0]: r['normalized_smiles'] for r in rows}

    def fake(spec, name, path, family, code):
        if family == 'pv':
            return np.stack([FA.pv_fingerprint(smi[float(c[0])], bits).astype(np.float32) * 2 - 1
                             for c in spec.collision_energy_ev]), None
        return np.stack([fp_of(smi[float(c[0])]).astype(np.float32) * 2 - 1 for c in spec.collision_energy_ev]), None

    models = [('full1', 'f1', 'fpnet', CODE), ('fpnet_0', 'f0', 'fpnet', CODE), ('fp_single_s2', 'ps', 'pv', None)]
    summ, dec = FA.run_a0(f'{d}/train.parquet', d, models, fake, k_other=2, n_control=1, out=f'{d}/a0', pv_bits=bits)
    s = summ.set_index(['qset', 'model'])
    check(s.loc[('np', 'fp_single_s2'), 'top1'] == 1.0 and s.loc[('np', 'fp_single_s2'), 'n'] == 2,
          'pv oracle ranks truth first, incl. a truth missing from the pool')
    check(dec == 'spectra', 'fpnet_0 == full1 on other-library spectra -> spectra only')


def test_worker_subprocess(d):
    # real subprocess workers on tiny random checkpoints: casmi.fpnet (A family) and prvsiyan pv_fp (P family)
    import torch
    rows = [dict(spec_row('gnps', ik(2), 'CCCO'), instrument_type='Orbitrap'),
            dict(spec_row('riken', ik(3), 'CC(C)O', ce=()), instrument_type=None)]
    spec = pd.DataFrame(rows)
    m = fpnet.FPNet(24, d=32, layers=1, heads=2)
    fpnet.save_ckpt(f'{d}/a.pt', m, 0)
    fn = FA.subprocess_logits_factory(d, pvfp=PVFP if os.path.exists(PVFP) else '')
    spec = spec.drop(columns=['ingest_lib']).assign(lib='gnps', qset='np')     # as run_a0 passes it
    z, err = fn(spec, 'tinyA', f'{d}/a.pt', 'fpnet', CODE)
    check(err is None and z.shape == (2, 24), f'fpnet worker logits {None if z is None else z.shape} {err}')
    if os.path.exists(PVFP):
        sys.path.insert(0, os.path.dirname(PVFP))
        import pv_fp
        net = pv_fp.FPNet(16, d=32, layers=1, heads=2)
        torch.save(dict(model=net.state_dict(), nbits=16, d=32, layers=1), f'{d}/fp_merged_m1.pt')
        z2, err2 = fn(spec, 'tinyP', f'{d}/fp_merged_m1.pt', 'pv', None)
        check(err2 is None and z2.shape == (2, 16) and np.abs(z2).sum() > 0, f'pv worker logits {err2}')
    else:
        print('  skip pv worker test: no pv_fp.py at', PVFP)
    z3, err3 = fn(spec, 'nope', f'{d}/missing.pt', 'fpnet', CODE)
    check(z3 is None and 'missing' in err3, 'missing checkpoint -> not scorable')


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2, \
            tempfile.TemporaryDirectory() as d3, \
            tempfile.TemporaryDirectory() as d4, tempfile.TemporaryDirectory() as d5:
        test_build_item(); test_sample(); test_read_rows(d1); test_window_rank(d2); test_verdict(); test_dry_run(d3)
        test_sample_np_sets(); test_a0_decision(); test_summarize_sets(); test_pv_fingerprint()
        test_inspect_npz(d4); test_a0_dry_run(d5)
    with tempfile.TemporaryDirectory() as d6:
        test_a0_pv_path(d6)
    with tempfile.TemporaryDirectory() as d7:
        test_worker_subprocess(d7)
    print(f'\n{"FAILED: " + str(len(FAIL)) if FAIL else "ALL PASS"} test_fold_audit')
    sys.exit(1 if FAIL else 0)
