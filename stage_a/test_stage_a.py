import os, sys, copy
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "build"))
import common, a3, a4
import isomer_mut as im
import iso_rank as ir

TYR = 'COc1ccc(CCN)cc1O'
SAL = 'OC(=O)c1ccccc1O'
FILL = ['C' * n + 'O' for n in range(1, 13)] + ['C' * n + '(=O)O' for n in range(1, 13)]


def spec_for(smi, mode=1):
    from rdkit import Chem
    from rdkit.Chem.Descriptors import ExactMolWt
    m = Chem.MolFromSmiles(smi); M = ExactMolWt(m) + 1.007276
    return [dict(mz=np.array([M - 18.0106, M - 17.0265, 77.04, 121.06, M]), it=np.array([.5, .3, .2, 1.0, .1]),
                 adduct='[M+H]+', prec=M, mode=mode)]


def test_gen_edges_matches_mutate_many_and_counts_multiplicity():
    parents = [TYR, SAL]
    ex = {im.ik14_plain(p) for p in parents}
    sel, pairs = common.gen_edges(parents, cap=60, deadline_s=30, exclude_keys=ex)
    ref = im.mutate_many(parents, cap=60, deadline=__import__('time').monotonic() + 30, exclude_keys=ex)
    assert [c[1] for c in sel] == [c[1] for c in ref] and len(sel) > 5
    assert all(k in pairs and len(pairs[k]) >= 1 for k in (c[1] for c in sel))


def test_split_deterministic_and_acyclic_by_parity():
    assert common.split_of('CCCCO', 'C3_00002') == 'A' and common.split_of('CCCCO', 'C3_00003') == 'B'
    s = {common.split_of(x, 'm1') for x in (TYR, SAL, 'c1ccccc1CCN', 'C1CCNCC1', 'c1ccc2ccccc2c1')}
    assert common.split_of(TYR, 'x') == common.split_of(TYR, 'y')
    assert common.scaffold_smiles(TYR) == 'c1ccccc1'


def test_a3_process_truth_free_record():
    a3.init()
    row = [TYR] + FILL[:24]
    r = a3.process(dict(mid='M1', row=row, spectra=spec_for(TYR), exclude=row, k=20))
    assert r['kids'] and len(r['kids']) <= 20 and r['parents'][0]['frag'] is not None
    assert all('nl' in k and 'pairs' in k and k['pairs'] for k in r['kids'])
    assert set(r['row']) <= set(r['keymap']) and all(k['smiles'] in r['keymap'] for k in r['kids'])
    assert r['parents'][0]['rank'] == 1 and len(r['parents']) == 10


def test_a3_frozen_mode_uses_given_kids():
    a3.init()
    row = [TYR] + FILL[:24]
    base = a3.process(dict(mid='M1', row=row, spectra=spec_for(TYR), exclude=row, k=5))
    froz = [(k['smiles'], k['ik14'], k['op'], 0.5) for k in base['kids']]
    r = a3.process(dict(mid='M1', row=row, spectra=spec_for(TYR), exclude=row, frozen=froz))
    assert [k['ik14'] for k in r['kids']] == [k['ik14'] for k in base['kids']] and r['stats']['frozen_without_edge'] == 0


def _tables(n=6):
    a3.init(); t, truth = {}, {}
    row = [TYR] + FILL[:24]
    for i in range(n):
        r = a3.process(dict(mid=f'C3_{i:05d}', row=row, spectra=spec_for(TYR), exclude=row, k=30))
        for k in r['kids']:
            k['ice'] = 0.3 + 0.01 * (hash(k['ik14']) % 50) / 50
        for p in r['parents']:
            p['ice'] = 0.2
        t[r['mid']] = r
    for m, r in t.items():
        truth[m] = r['keymap'][r['kids'][3]['smiles']][1]      # a child is the "truth"
    return t, truth


def test_replay_and_worst_case():
    t, truth = _tables(2)
    a4.patch_keys({k: v for r in t.values() for k, v in r['keymap'].items()})
    m = next(iter(t))
    x = a4.replay_one(t[m], truth[m], dict(ISO_M_ICE=0.0, ISO_LISTS=a4.CAND_LISTS))
    assert x['rr0'] == 0 and x['rr_wc'] == 0                      # truth is a child: absent from V22, never counted worst-case
    assert x['has_kids']
    y = a4.replay_one(t[m], truth[m], dict(ISO_M_ICE=0.0, ISO_LISTS=a4.CAND_LISTS), frag_only=True)
    assert 'fired' in y


def test_freeze_never_reads_half_b():
    t, truth = _tables(6)
    a4.patch_keys({k: v for r in t.values() for k, v in r['keymap'].items()})
    mids = list(t)
    a_mids, b_mids = mids[:3], mids[3:]

    class Guard(dict):
        def __getitem__(self, k):
            assert k in a_mids, "half B truth touched"
            return dict.__getitem__(self, k)
    ta = Guard({m: truth[m] for m in a_mids})
    fz = a4.freeze(t, ta, a_mids)
    assert fz['lists'] and fz['m_ice'] in a4.M_GRID and set(fz['grid_delta_A']) == {str(m) for m in a4.M_GRID}
    assert fz['pool_mrr20_A'] > 0


def test_composite_and_boot():
    assert abs(a4.composite(0.0, 0.0, 0.02, 0.25) - 0.005) < 1e-12
    m, lo, hi = a4.boot([0.1, 0.0, 0.2, 0.0], ['a', 'a', 'b', 'c'])
    assert lo <= m <= hi


def test_forward_select_keeps_only_rising_lists():
    t, truth = _tables(4)
    a4.patch_keys({k: v for r in t.values() for k, v in r['keymap'].items()})
    lists, best, trace = a4.forward_select(t, truth, list(t))
    assert 1 <= len(lists) <= 5 and len(set(lists)) == len(lists)
