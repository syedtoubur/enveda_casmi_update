"""Unit tests for build/iso_nl.py (neutral-loss + 6-ring RDA scorer, measured spectrum only)."""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import iso_nl  # noqa: E402

SAL = 'OC(=O)c1ccccc1O'        # salicylic acid
P4 = 'OC(=O)c1ccc(O)cc1'       # 4-hydroxybenzoic acid
ETHER = 'COc1ccccc1C=O'        # 2-methoxybenzaldehyde, no carboxylic acid


def spec(prec, losses, adduct='[M+H]+'):
    mz = np.array([prec - x for x in losses] + [prec], np.float64)
    it = np.array([1.0] * len(losses) + [0.2], np.float64)
    return dict(mz=mz, it=it, adduct=adduct, prec=prec, mode=1)


def test_diagnostic_loss_separates_candidates():
    prec = 139.0390                                  # [M+H]+ of C7H6O3
    s = spec(prec, [18.010565, 43.989829])           # H2O and CO2 losses
    sc = iso_nl.nl_scores([s], [SAL, P4, ETHER, 'not a smiles'])
    assert sc.shape == (4,) and np.isnan(sc[3])
    assert sc[0] > 0 and sc[1] > 0                   # both acids carry OH and COOH
    assert sc[0] > sc[2] and sc[1] > sc[2]           # the acids beat the loss-free ether
    assert abs(sc[0] - sc[1]) < 1e-9                 # the loss rules do not separate these two regioisomers


def test_regioisomers_differ_on_a_ring_rda_peak():
    a, b = 'C1=CCCC(C)C1', 'CC1=CCCCC1'                # 4- vs 1-methylcyclohexene: different RDA pieces
    fa = iso_nl.rda_fragments(a)
    fb = iso_nl.rda_fragments(b)
    assert len(fa) > 0 and len(fb) > 0
    only_a = sorted(set(np.round(fa, 4)) - set(np.round(fb, 4)))
    assert only_a, 'fixture must give a diagnostic RDA mass'
    s = dict(mz=np.array([only_a[0] + iso_nl.PROTON]), it=np.array([1.0]), adduct='[M+H]+', prec=99.1168, mode=1)
    sc = iso_nl.nl_scores([s], [a, b])
    assert sc[0] > sc[1] >= 0


def test_empty_spectra_and_unparseable():
    assert list(iso_nl.nl_scores([], [SAL, P4])) == [0.0, 0.0]
    out = iso_nl.nl_scores([spec(139.039, [18.010565])], [''])
    assert out.shape == (1,) and np.isnan(out[0])
