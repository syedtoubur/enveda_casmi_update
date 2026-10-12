"""Tests for smart_gate.py. Run: python .scratch/smart_gate/test_smart_gate.py (pytest also works)."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import smart_gate as SG  # noqa: E402

EVAL_NB = HERE.parent / 'kaggle_libmax_eval' / 'casmi_cross_instrument_libmax_eval.ipynb'
LIBS = ['enveda-180', 'pluskal_ms2', 'riken', 'gnps', 'massbank', 'mona', 'spectraverse',
        'msdial', 'drug_plus', 'enveda-np-examples', 'masaryk']


# ------------------------------------------------------------------ fixtures
def rand_spec(rng, prec, n=30):
    mz = np.sort(rng.uniform(50, prec + 1, n))
    it = rng.uniform(0, 1, n)
    return mz, it


class FakeLib:
    """Minimal casmi.library.Library: CSR peaks + per-spectrum metadata; records window() args."""

    def __init__(self, spectra, sid, lib, mode, adduct_ix, smiles):
        self.calls = []
        off, mz, it = [0], [], []
        for m, i in spectra:
            mz.extend(m); it.extend(i); off.append(len(mz))
        self.off = np.asarray(off, np.int64)
        self.mz = np.asarray(mz, np.float32)
        self.it = np.asarray(it, np.float32)
        self.sid = np.asarray(sid, np.int64)
        self.lib = np.asarray(lib, np.int8)
        self.mode = np.asarray(mode, np.int8)
        self.adduct_ix = np.asarray(adduct_ix, np.int16)
        self.struct_smiles = np.asarray(smiles, dtype=object)

    def window(self, target, ppm, exclude=None, mode=None):
        self.calls.append((target, ppm, mode))
        c = np.arange(len(self.sid))
        return c[self.mode[c] == mode] if mode is not None else c


ADD = {'[M+H]+': 1, '[M+Na]+': 2, '[M-H]-': 3}
adduct_index = ADD.get


def query(mz, it, prec=400.0, mode=1, adduct='[M+H]+'):
    return dict(mz=np.asarray(mz, np.float64), it=np.asarray(it, np.float64), prec=prec, mode=mode, adduct=adduct)


def frag_spec(prec=400.0):
    """Fragment-rich spectrum with peaks above prec/2."""
    mz = np.array([80.05, 120.07, 150.1, 210.12, 250.13, 300.15, 330.2, prec])
    it = np.array([0.3, 0.5, 0.2, 0.6, 1.0, 0.4, 0.7, 0.9])
    return mz, it


# ------------------------------------------------------------------ 1. parity with Phase-3 eval cell 5
def test_parity_with_eval_cell5():
    nb = json.loads(EVAL_NB.read_text(encoding='utf-8'))
    src = next(''.join(c['source']) for c in nb['cells']
               if c['cell_type'] == 'code' and 'def compare_spectra' in ''.join(c['source']))
    ns = {'np': np}
    exec(src, ns)
    rng = np.random.default_rng(0)
    for k in range(200):
        prec = float(rng.uniform(150, 900))
        a, b = rand_spec(rng, prec, int(rng.integers(1, 90))), rand_spec(rng, prec, int(rng.integers(1, 90)))
        if k % 3 == 0:       # force shared peaks so the match path is exercised
            b = (np.sort(np.concatenate([a[0][: len(a[0]) // 2] * (1 + 3e-6), b[0]])),
                 np.concatenate([a[1][: len(a[1]) // 2], b[1]])[np.argsort(np.concatenate([a[0][: len(a[0]) // 2] * (1 + 3e-6), b[0]]))])
        for mask in (True, False):
            p1 = SG.clean_and_prep_spectrum(*a, prec_mz=prec, mask_precursor=mask)
            e1 = ns['clean_and_prep_spectrum'](*a, prec_mz=prec, mask_precursor=mask)
            p2 = SG.clean_and_prep_spectrum(*b, prec_mz=prec, mask_precursor=mask)
            e2 = ns['clean_and_prep_spectrum'](*b, prec_mz=prec, mask_precursor=mask)
            assert SG.compare_spectra(p1, p2, prec) == ns['compare_spectra'](e1, e2, prec)


# ------------------------------------------------------------------ 2. window is called in ppm, centred
def test_window_called_with_ppm_and_centre():
    m, i = frag_spec()
    L = FakeLib([(m, i)], [0], [3], [1], [1], ['CCO'])
    SG.evaluate_molecule(L, [query(m, i)], 500.0, [0, 9], adduct_index, lambda s: f'K{s}')
    target, ppm, mode = L.calls[0]
    assert ppm == 10.0 and mode == 1
    assert abs(target - 500.0 * (1 - 1.45e-6)) < 1e-9


# ------------------------------------------------------------------ 3. tiers, filters, support, margin
def test_identical_spectrum_is_tier1_and_filters_apply():
    m, i = frag_spec()
    # ref 0: identical (gnps) -> T1. ref 1: identical but enveda-180 -> excluded.
    # ref 2: other polarity -> excluded by mode. ref 3: sid<0 -> excluded. ref 4: other adduct -> excluded.
    L = FakeLib([(m, i)] * 5, [0, 1, 2, -1, 4], [3, 0, 3, 3, 3], [1, 1, -1, 1, 1], [1, 1, 1, 1, 2],
                ['C0', 'C1', 'C2', 'Cx', 'C4'])
    d = SG.evaluate_molecule(L, [query(m, i)], 500.0, [0, 9], adduct_index, lambda s: f'K{s}')
    assert d['tier'] == 1 and d['sid'] == 0 and d['smiles'] == 'C0' and d['S'] > 0.99
    assert d['rival'] == 0.0 and d['n_ref'] == 1


def test_few_matched_peaks_never_lock():
    mz = np.array([100.0, 250.0, 300.0, 400.0]); it = np.array([1.0, 0.5, 0.5, 1.0])   # 3 fragments after masking
    L = FakeLib([(mz, it)], [0], [3], [1], [1], ['C0'])
    d = SG.evaluate_molecule(L, [query(mz, it)], 500.0, [0], adduct_index, lambda s: f'K{s}')
    assert d['tier'] == 4


def test_decide_tier_boundaries():
    assert SG.decide_tier(0.85, 0.0, 1, 1) == (1, False)
    assert SG.decide_tier(0.8499, 0.0, 1, 1) == (2, False)
    assert SG.decide_tier(0.65, 0.0, 1, 1) == (2, False)
    assert SG.decide_tier(0.6499, 0.0, 1, 1) == (3, False)        # one spectrum: support 1 is enough
    assert SG.decide_tier(0.50, 0.0, 1, 3) == (4, False)          # 3 spectra: T3 needs support >= 2
    assert SG.decide_tier(0.50, 0.0, 2, 3) == (3, False)
    assert SG.decide_tier(0.3999, 0.0, 5, 5) == (4, False)
    assert SG.decide_tier(0.90, 0.86, 1, 1) == (3, True)          # rival within margin: T1 -> T3
    assert SG.decide_tier(0.70, 0.66, 1, 1) == (3, True)
    assert SG.decide_tier(0.50, 0.46, 2, 2) == (4, True)          # T3 -> T4
    assert SG.decide_tier(0.90, 0.85, 1, 1) == (1, False)         # margin exactly 0.05 keeps the tier


def test_rival_counted_per_identity_not_per_sid():
    m, i = frag_spec()
    L = FakeLib([(m, i), (m, i)], [0, 1], [3, 3], [1, 1], [1, 1], ['C0', 'C1'])
    same = SG.evaluate_molecule(L, [query(m, i)], 500.0, [0], adduct_index, lambda s: 'SAME')
    assert same['tier'] == 1 and same['rival'] == 0.0             # two sids, one identity: no rival
    diff = SG.evaluate_molecule(L, [query(m, i)], 500.0, [0], adduct_index, lambda s: f'K{s}')
    assert diff['tier'] == 3 and diff['margin_demoted']            # isomer twin: demoted


def test_run_all_codes_budget_and_dict_spectra():
    import pandas as pd
    m, i = frag_spec()
    L = FakeLib([(m, i)], [0], [3], [1], [1], ['C0'])
    rows = [dict(molecule_id='a', ms2_mzs=m, ms2_normalized_intensities=i, ionization_mode='positive',
                 adduct='[M+H]+', precursor_mz=400.0, nm=500.0),
            dict(molecule_id='b', ms2_mzs=m, ms2_normalized_intensities=i, ionization_mode='positive',
                 adduct='[M+H]+', precursor_mz=400.0, nm=np.nan)]
    te = pd.DataFrame(rows)
    mols = list(te.groupby('molecule_id', sort=False))
    gates, summ = SG.run_all(L, mols, LIBS, adduct_index, lambda s: f'K{s}', log_every=0)
    assert gates['a']['tier'] == 1 and gates['b']['tier'] == 4
    assert summ['health'] == 'ok' and summ['excl_lib_ix'] == [0, 9] and summ['tiers']['1'] == 1
    json.dumps(summ, allow_nan=False)
    json.dumps(gates, allow_nan=False)
    gates, summ = SG.run_all(L, mols, LIBS, adduct_index, lambda s: f'K{s}', budget_s=-1, log_every=0)
    assert gates == {} and summ['timed_out'] == 2 and summ['health'].startswith('partial')


# ------------------------------------------------------------------ 4. Cell-22 merge equivalence
S_TAU, POP_TAU, LIB_TAU, REL_TH = 6.0, 5.0, 0.9, 600.0
SLOTS_AGG, SLOTS_GENTLE = [2, 4, 6, 8, 10], [4, 8, 12, 16, 20]


def merge(base, base_keys, pc, pc_keys, slots, n=25):          # verbatim from the notebook
    bk = set(base_keys)
    pcs = [s for s, k in zip(pc, pc_keys) if k not in bk]
    out, bi, pj = [], 0, 0
    for pos in range(1, n + 1):
        if pos in slots and pj < len(pcs):
            out.append(pcs[pj]); pj += 1
        elif bi < len(base):
            out.append(base[bi]); bi += 1
        elif pj < len(pcs):
            out.append(pcs[pj]); pj += 1
    return out


def promote(base, base_keys, pc, pc_keys, slots, n=25):        # verbatim from the notebook
    usual = merge(base, base_keys, pc, pc_keys, slots, n=n)
    if not pc:
        return usual
    top = pc[0]
    return ([top] + [x for x in usual if x != top])[:n]


def old_cell22(smis, keys, lib_max, p, stats):                   # verbatim loop body from the notebook
    if p is None or not p['pc']:
        stats['no_pc'] += 1; final = smis
    elif lib_max >= LIB_TAU:
        stats['untouched'] += 1; final = smis
    else:
        rel = (p['fz_top'] if p.get('fz_top') is not None else p['pc_fz'][0]) - p['best_pool_fz'] if np.isfinite(p['best_pool_fz']) else 1e9
        slots = SLOTS_AGG if rel > REL_TH else SLOTS_GENTLE
        stats['aggressive' if rel > REL_TH else 'gentle'] += 1
        S = p.get('S'); pop_top = float(p.get('top_pop') or 0.0)
        if (S is not None and float(S) > S_TAU and pop_top >= POP_TAU) or (rel > 250.0):
            stats['promoted'] += 1
            final = promote(smis, keys, p['pc'], p['pc_keys'], slots)
        else:
            final = merge(smis, keys, p['pc'], p['pc_keys'], slots)
    return final


def rand_case(rng):
    nb, npc = int(rng.integers(0, 30)), int(rng.integers(0, 30))
    smis = [f'b{j}' for j in range(nb)]; keys = [f'kb{j}' for j in range(nb)]
    pcs = [f'p{j}' if rng.random() > 0.2 else f'b{j}' for j in range(npc)]
    p = None if rng.random() < 0.1 else dict(
        pc=pcs, pc_keys=['k' + s for s in pcs], pc_fz=list(rng.normal(0, 500, npc)) or [0.0],
        fz_top=None if rng.random() < 0.5 else float(rng.normal(0, 500)),
        best_pool_fz=float('nan') if rng.random() < 0.1 else float(rng.normal(-500, 400)),
        S=None if rng.random() < 0.2 else float(rng.normal(6, 2)), top_pop=float(rng.uniform(0, 10)))
    return smis, keys, float(rng.uniform(0.5, 1.0)), p


KW = dict(LIB_TAU=LIB_TAU, REL_TH=REL_TH, SLOTS_AGG=SLOTS_AGG, SLOTS_GENTLE=SLOTS_GENTLE, S_TAU=S_TAU,
          POP_TAU=POP_TAU, merge=merge, promote=promote)


def test_tier4_and_none_reproduce_old_cell22():
    rng = np.random.default_rng(1)
    for _ in range(3000):
        smis, keys, lib_max, p = rand_case(rng)
        for dec in (None, dict(tier=4)):
            st = dict(untouched=0, gentle=0, aggressive=0, no_pc=0, promoted=0)
            old = old_cell22(smis, keys, lib_max, p, st)
            new, labels = SG.merge_decision(smis, keys, lib_max, p, dec, **KW)
            assert new == old
            assert {k: v for k, v in st.items() if v} == {l: 1 for l in labels if l in st}


def test_superset_and_tiers_never_add_pubchem_where_old_did_not():
    rng = np.random.default_rng(2)
    for _ in range(3000):
        smis, keys, lib_max, p = rand_case(rng)
        old = old_cell22(smis, keys, lib_max, p, dict(untouched=0, gentle=0, aggressive=0, no_pc=0, promoted=0))
        for t in (1, 2, 3):
            new, labels = SG.merge_decision(smis, keys, lib_max, p, dict(tier=t), **KW)
            if old == smis:                    # old left it untouched -> new must too
                assert new == smis
            if t in (1, 2):
                assert new == smis
            if t == 3 and p and p['pc'] and lib_max < LIB_TAU:
                assert labels == ['t3'] and new == merge(smis, keys, p['pc'], p['pc_keys'], SG.SLOTS_T3)
                assert new[:1] == smis[:1] or not smis   # no promotion over the library list


def test_tier3_first_molecule_has_no_stale_rel():
    p = dict(pc=['x'], pc_keys=['kx'], pc_fz=[1.0], best_pool_fz=0.0)   # no fz_top, no S
    new, labels = SG.merge_decision(['a'], ['ka'], 0.1, p, dict(tier=3), **KW)
    assert labels == ['t3'] and new == ['a', 'x']


# ------------------------------------------------------------------ 5. apply_lock
def safe_extend(original, extra, limit=25):                    # notebook semantics, identity = string
    out, seen = [], set()
    for smi in list(original) + list(extra):
        if smi not in seen:
            seen.add(smi); out.append(smi)
            if len(out) == limit:
                break
    return out


ident = lambda s: s.lower() if s else None


def test_apply_lock_cases():
    res = [f's{j}' for j in range(25)]
    assert SG.apply_lock(res, None, ident, safe_extend) == (res, None)
    assert SG.apply_lock(res, dict(tier=4, smiles='s3'), ident, safe_extend) == (res, None)
    out, ev = SG.apply_lock(res, dict(tier=1, smiles='S7'), ident, safe_extend)   # identity match, not string
    assert ev == 'pinned_moved' and out[0] == 's7' and len(out) == 25 and len(set(out)) == 25
    assert out[1:8] == res[:7] and out[8:] == res[8:]
    out, ev = SG.apply_lock(res, dict(tier=2, smiles='new'), ident, safe_extend)
    assert ev == 'inserted' and out[0] == 'new' and len(out) == 25 and out[1:] == res[:24]
    out, ev = SG.apply_lock(res, dict(tier=3, smiles='s9'), ident, safe_extend)
    assert ev == 'top2_moved' and out[:3] == ['s0', 's9', 's1']
    assert SG.apply_lock(res, dict(tier=3, smiles='s1'), ident, safe_extend) == (res, 'kept')
    assert SG.apply_lock(res, dict(tier=1, smiles='s0'), ident, safe_extend) == (res, 'kept')
    out, ev = SG.apply_lock(['a'], dict(tier=3, smiles='z'), ident, safe_extend)
    assert ev == 'inserted' and out == ['a', 'z']
    out, ev = SG.apply_lock([], dict(tier=1, smiles='z'), ident, safe_extend)
    assert ev == 'inserted' and out == ['z']
    assert SG.apply_lock(res, dict(tier=1, smiles=''), ident, safe_extend) == (res, None)
    assert SG.apply_lock(res, dict(tier=1, smiles='bad'), lambda s: None, safe_extend) == (res, 'identity_fail')


def test_locked():
    assert SG.locked(dict(tier=1)) and SG.locked(dict(tier=2))
    assert not SG.locked(dict(tier=3)) and not SG.locked(None) and not SG.locked({})


def test_t3_switch_off_acts_like_t4():
    rng = np.random.default_rng(3)
    SG.T3_ACTIVE = False
    try:
        for _ in range(500):
            smis, keys, lib_max, p = rand_case(rng)
            assert SG.merge_decision(smis, keys, lib_max, p, dict(tier=3), **KW) ==                 SG.merge_decision(smis, keys, lib_max, p, dict(tier=4), **KW)
        res = [f's{j}' for j in range(5)]
        assert SG.apply_lock(res, dict(tier=3, smiles='s4'), ident, safe_extend) == (res, None)
        assert SG.apply_lock(res, dict(tier=1, smiles='s4'), ident, safe_extend)[1] == 'pinned_moved'
    finally:
        SG.T3_ACTIVE = True


if __name__ == '__main__':
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn(); print('PASS', name)
            except Exception as e:  # noqa: BLE001
                fails += 1; print('FAIL', name, repr(e))
                import traceback; traceback.print_exc()
    print('failures', fails)
    sys.exit(1 if fails else 0)
