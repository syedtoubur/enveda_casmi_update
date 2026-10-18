"""Unit tests for build/iso_rank.py and the ISO_MUT knob of build/casmi_post.py (real SMILES only)."""
from __future__ import annotations

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import casmi_post as cp  # noqa: E402
import iso_rank as ir  # noqa: E402
from isomer_mut import ik14_plain, mutate  # noqa: E402

SAL = 'OC(=O)c1ccccc1O'          # salicylic acid (parent)
P4 = 'OC(=O)c1ccc(O)cc1'         # 4-hydroxybenzoic acid
P3 = 'OC(=O)c1cccc(O)c1'         # 3-hydroxybenzoic acid
TYR = 'COc1ccc(CCN)cc1O'          # a real parent with many isomer children
RES = 'CC(=O)Nc1ccc(O)cc1'       # paracetamol (second parent)
RES_C = 'CC(=O)Nc1cccc(O)c1'     # its meta isomer
FILL = ['C' * n + 'O' for n in range(1, 13)] + ['C' * n + '(=O)O' for n in range(1, 13)]


def edge(smi, parent, rank, op='swap'):
    return dict(smiles=smi, ik14=ik14_plain(smi), parent_ik14=ik14_plain(parent), parent_rank=rank, op=op)


def row25(parent_at=0, parent=SAL):
    r = list(FILL[:24])
    r.insert(parent_at, parent)
    return r[:25]


def rec_of(children, ice, frag, pice, pfrag, parent=SAL):
    sc = dict(ice={ik14_plain(s): v for s, v in ice.items()}, frag={ik14_plain(s): v for s, v in frag.items()})
    return dict(children=children, scores=sc, parent_scores={ik14_plain(parent): dict(ice=pice, frag=pfrag)})


def test_fixture_is_not_vacuous():
    r = row25(0)
    assert len(r) == 25 and len(set(r)) == 25 and r[0] == SAL
    assert len({ik14_plain(s) for s in r}) == 25
    assert ik14_plain(P4) and len({ik14_plain(P4), ik14_plain(P3), ik14_plain(SAL)}) == 3
    assert len(mutate(SAL, cap=50)) > 0


def test_parent_map_multiplicity_and_best_parent():
    kids = [edge(P4, SAL, 3, 'a'), edge(P4, SAL, 3, 'b'), edge(P4, RES, 1, 'a'), edge(P4, RES, 1, 'a'), edge(P3, SAL, 3, 'a')]
    m = ir.parent_map(kids)
    assert set(m) == {ik14_plain(P4), ik14_plain(P3)}
    r = m[ik14_plain(P4)]
    assert r['multiplicity'] == 3                       # (SAL,a) (SAL,b) (RES,a)
    assert r['parent_rank'] == 1 and r['parent_ik14'] == ik14_plain(RES)
    assert m[ik14_plain(P3)]['multiplicity'] == 1 and m[ik14_plain(P3)]['smiles'] == P3


def test_real_mutate_children_feed_parent_map():
    ch = [dict(smiles=s, ik14=ik14_plain(s), parent_ik14=ik14_plain(SAL), parent_rank=1, op=op)
          for s, op, _ in mutate(SAL, cap=40)]
    m = ir.parent_map(ch)
    assert m and all(v['parent_ik14'] == ik14_plain(SAL) for v in m.values())
    assert all(v['multiplicity'] >= 1 for v in m.values())


def test_rrf_missing_lists_ties_and_direction():
    recs = ir.parent_map([edge(P4, SAL, 1), edge(P3, SAL, 2)])
    a, b = ik14_plain(P4), ik14_plain(P3)
    # parent_rank is lower-better: a (rank 1) first; derived from the records
    assert ir.rrf(recs, {}, lists=('parent_rank',))[0] == a
    # ICE is higher-better
    assert ir.rrf(recs, {'ice': {a: 0.1, b: 0.9}}, lists=('ice',))[0] == b
    # a list missing for b is skipped for b only
    assert ir.rrf(recs, {'nl': {a: 0.5}}, lists=('nl',)) == [a, b]
    # a tie gives equal scores -> stable tie-break by ik14
    assert ir.rrf(recs, {'ice': {a: 0.5, b: 0.5}}, lists=('ice',)) == sorted([a, b])
    # NaN counts as missing
    assert ir.rrf(recs, {'ice': {a: float('nan'), b: 0.2}}, lists=('ice',)) == [b, a]


def test_rrf_average_rank_on_ties_and_k():
    three = ir.parent_map([edge(P4, SAL, 1), edge(P3, SAL, 1), edge(RES_C, RES, 1)])
    ks = list(three)
    vals = {ks[0]: 0.9, ks[1]: 0.1, ks[2]: 0.1}           # ranks 1, 2.5, 2.5
    order = ir.rrf(three, {'ice': vals}, lists=('ice',))
    assert order[0] == ks[0] and order[1:] == sorted(ks[1:])
    # two lists that disagree: score = 1/(k+r1) + 1/(k+r2); k=10, ranks (1,3) vs (3,1) tie -> ik14 order
    x = {ks[0]: 3, ks[1]: 2, ks[2]: 1}
    y = {ks[0]: 1, ks[1]: 2, ks[2]: 3}
    o = ir.rrf(three, {'ice': x, 'frag': y}, k=10, lists=('ice', 'frag'))
    assert o[0] == ks[1] or set(o[:2]) == {ks[0], ks[2]} or True
    s = {k_: 1 / (10 + r1) + 1 / (10 + r2) for k_, r1, r2 in ((ks[0], 1, 3), (ks[1], 2, 2), (ks[2], 3, 1))}
    assert o == sorted(ks, key=lambda k_: (-s[k_], k_))


def test_fire_places_below_parent_rank1_goes_to_rank2():
    rec = rec_of([edge(P4, SAL, 1)], {P4: 0.8}, {P4: 0.7}, 0.5, 0.4)
    row = row25(0)
    new, info = ir.apply(row, rec, dict(ISO_TAIL=False))
    assert info == dict(fired=True, tail=0)
    assert new[0] == SAL and new[1] == P4 and len(new) == 25
    assert new[2:] == row[1:24]                         # the rest shifted down by one, last dropped


def test_placement_below_deeper_parent():
    rec = rec_of([edge(P4, SAL, 1)], {P4: 0.8}, {P4: 0.7}, 0.5, 0.4)
    row = row25(6)
    new, info = ir.apply(row, rec, dict(ISO_TAIL=False))
    assert info['fired'] and new[6] == SAL and new[7] == P4 and new[8:] == row[7:24]


def test_no_fire_cases():
    kids = [edge(P4, SAL, 1)]
    cfg = dict(ISO_TAIL=False)
    row = row25(0)
    for ice, frag, pice, pfrag in [(0.4, 0.7, 0.5, 0.4),      # ICE lower
                                   (0.8, 0.3, 0.5, 0.4),      # frag lower
                                   (0.5, 0.7, 0.5, 0.4)]:     # ICE equal: strict inequality
        new, info = ir.apply(row, rec_of(kids, {P4: ice}, {P4: frag}, pice, pfrag), cfg)
        assert not info['fired'] and new == row
    assert not ir.apply(row, rec_of(kids, {}, {P4: 0.7}, 0.5, 0.4), cfg)[1]['fired']          # child ICE missing
    assert not ir.apply(row, rec_of(kids, {P4: 0.8}, {P4: 0.7}, None, 0.4), cfg)[1]['fired']  # parent ICE missing
    assert not ir.apply(row, rec_of(kids, {P4: 0.8}, {}, 0.5, 0.4), cfg)[1]['fired']          # child frag missing
    sc = rec_of(kids, {P4: 0.8}, {P4: 0.7}, 0.5, 0.4)
    gone = [r for r in row if r != SAL] + ['CCCCCCCCCCCCCCCO']                               # parent absent
    new, info = ir.apply(gone, sc, cfg)
    assert not info['fired'] and new == gone
    assert not ir.apply(row, sc, dict(cfg, ISO_M_ICE=0.5))[1]['fired']                       # margin 0.3 < 0.5
    assert ir.apply(row, sc, dict(cfg, ISO_M_ICE=0.2))[1]['fired']


def test_dedupe_child_already_in_row_next_child_used():
    kids = [edge(P4, SAL, 1), edge(P3, SAL, 1)]
    rec = rec_of(kids, {P4: 0.9, P3: 0.8}, {P4: 0.9, P3: 0.8}, 0.5, 0.4)
    row = row25(0)
    row[10] = P4                                         # best child already retrieved
    new, info = ir.apply(row, rec, dict(ISO_TAIL=False))
    assert info['fired'] and new[1] == P3
    assert new.count(P4) == 1 and len({ik14_plain(s) for s in new}) == 25


def test_dedupe_other_spelling_of_child_is_matched():
    rec = rec_of([edge(P4, SAL, 1)], {P4: 0.8}, {P4: 0.7}, 0.5, 0.4)
    row = row25(0)
    row[5] = 'Oc1ccc(cc1)C(O)=O'                         # 4-hydroxybenzoic acid, other spelling
    new, info = ir.apply(row, rec, dict(ISO_TAIL=False))
    assert not info['fired'] and new == row


def test_cut_at_25_and_parent_at_last_rank():
    rec = rec_of([edge(P4, SAL, 1)], {P4: 0.8}, {P4: 0.7}, 0.5, 0.4)
    new, info = ir.apply(row25(24), rec, dict(ISO_TAIL=False))
    assert not info['fired'] and len(new) == 25 and new[24] == SAL
    new, info = ir.apply(row25(23), rec, dict(ISO_TAIL=False))
    assert info['fired'] and new[24] == P4 and len(new) == 25 and new[23] == SAL


def test_tail_fills_ranks_21_to_25_with_next_rrf_children():
    ms = [s for s, _, _ in mutate(TYR, cap=60)]
    kids = []
    for i, s in enumerate(ms):
        e = edge(s, TYR, 1, 'op%d' % i)
        if e['ik14'] and e['ik14'] not in {k['ik14'] for k in kids}:
            kids.append(e)
    assert len(kids) >= 6
    row = row25(0, parent=TYR)
    rec = dict(children=kids, scores={}, parent_scores={})        # no ICE -> no fire, the tail still rides along
    new, info = ir.apply(row, rec, dict(ISO_TAIL=True))
    assert not info['fired'] and info['tail'] == 5 and len(new) == 25
    assert new[:20] == row[:20]
    kid_keys = {k['ik14'] for k in kids}
    assert {ik14_plain(s) for s in new[20:]} <= kid_keys
    assert len({ik14_plain(s) for s in new}) == 25
    assert ir.apply(row, rec, dict(ISO_TAIL=False))[0] == row


def test_tail_with_fewer_children_than_slots_pads_with_originals():
    rec = dict(children=[edge(P4, SAL, 1), edge(P3, SAL, 1)], scores={}, parent_scores={})
    row = row25(0)
    new, info = ir.apply(row, rec, dict(ISO_TAIL=True))
    assert info['tail'] == 2 and len(new) == 25
    assert set(new[20:22]) == {P4, P3} and new[22:] == row[20:23]


def test_tail_plus_fire_keeps_inserted_child():
    rec = rec_of([edge(P4, SAL, 1), edge(P3, SAL, 2)], {P4: 0.9, P3: 0.1}, {P4: 0.9, P3: 0.1}, 0.5, 0.4)
    row = row25(0)
    new, info = ir.apply(row, rec, dict(ISO_TAIL=True))
    assert info['fired'] and new[1] == P4 and info['tail'] == 1
    assert P3 in new[20:] and new.count(P4) == 1 and len({ik14_plain(s) for s in new}) == 25


def test_fire_at_tail_zone_slot_is_not_overwritten():
    rec = rec_of([edge(P4, SAL, 1), edge(P3, SAL, 2)], {P4: 0.9, P3: 0.1}, {P4: 0.9, P3: 0.1}, 0.5, 0.4)
    row = row25(20)
    new, info = ir.apply(row, rec, dict(ISO_TAIL=True))
    assert info['fired'] and new[20] == SAL and new[21] == P4 and P3 in new


def test_iso_lists_subset_changes_order():
    recs = ir.parent_map([edge(P4, SAL, 2), edge(P3, SAL, 1)])
    sc = {'ice': {ik14_plain(P4): 0.9, ik14_plain(P3): 0.1}}
    assert ir.rrf(recs, sc, lists=('ice',))[0] == ik14_plain(P4)
    assert ir.rrf(recs, sc, lists=('parent_rank',))[0] == ik14_plain(P3)


# ------------------------------------------------------------------ casmi_post knob
def _state():
    import test_casmi_post as t
    return t, t.state()


def test_iso_mut_off_is_byte_identical():
    t, s = _state()
    base = cp.run(s, t.deps(), stats={})
    s2 = dict(s, iso={'m1': dict(children=[edge(P4, SAL, 1)], scores={}, parent_scores={})})
    assert cp.run(s2, t.deps(), stats={}) == base                 # knob off: state['iso'] is ignored
    st = {}
    cp.run(s, t.deps(), stats=st)
    assert not any(k.startswith('iso_') for k in st)
    assert cp.CFG['ISO_MUT'] is False
    assert cp.run(s, t.deps(), stats={}, ISO_MUT=True) == base    # on, but no state['iso']: unchanged


def test_iso_mut_on_applies_and_counts():
    t, s = _state()
    base = cp.run(s, t.deps(), stats={})
    parent = base['m1'][0]
    child = 'CC(C)CCO'                                            # real SMILES (isopentanol), not in the row
    assert child not in base['m1'] and len(base['m1']) >= 3
    rec = dict(children=[edge(child, parent, 1)],
               scores=dict(ice={ik14_plain(child): 0.9}, frag={ik14_plain(child): 0.9}),
               parent_scores={ik14_plain(parent): dict(ice=0.1, frag=0.1)})
    s2 = dict(s, iso={'m1': rec, 'm2': dict(children=[], scores={}, parent_scores={})})
    st = {}
    out = cp.run(s2, t.deps(), stats=st, ISO_MUT=True, ISO_TAIL=False)
    assert out['m1'][0] == parent and out['m1'][1] == child
    assert out['m2'] == base['m2'] and out['m3'] == base['m3']
    assert st['iso_children'] == 1 and st['iso_fired'] == 1 and st['iso_tail'] == 0


def test_truth_blind_audit():
    pat = re.compile(r"truth|ik14_true|val_key|['\"]inchikey['\"]", re.I)
    for fn in ('iso_rank.py', 'iso_nl.py'):
        with open(os.path.join(HERE, fn), encoding='utf-8') as f:
            txt = f.read()
        hits = [ln for ln in txt.splitlines() if pat.search(ln)]
        assert not hits, (fn, hits)
        assert not re.search(r"\bopen\(|read_parquet|read_csv|pickle|np\.load", txt), fn
