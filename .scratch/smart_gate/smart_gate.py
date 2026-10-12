"""Smart library gate for casmi26-e006 (plan: .scratch/smart_gate/PLAN.md).

Single source of truth: build_nb.py embeds this file's text into the notebook, and
test_smart_gate.py imports it. Pure module: numpy only, no notebook globals, no RDKit.

Score = Phase-3 masked spectral entropy (clean_and_prep_spectrum + compare_spectra, copied
verbatim from .scratch/kaggle_libmax_eval/casmi_cross_instrument_libmax_eval.ipynb cell 5),
so the tier cut-offs mean what they meant on the Phase-3 operating curve.
"""
import time

import numpy as np

EXCL_LIBS = ('enveda-180', 'enveda-np-examples')   # Phase-3 windows held non-timsTOF libraries only
PPM_WIN, PPM_CENTER = 10.0, 1.45                    # L.window takes ppm; timsTOF +1.45 ppm offset
TAU1, TAU2, TAU3 = 0.85, 0.65, 0.40
N_MIN = 4                                           # Phase-3 "Smart" filter: n_matched >= 4 and has_high_mz
MARGIN = 0.05                                       # uncalibrated: rival identity within MARGIN -> demote one tier
SLOTS_T3 = [15, 18, 21]
T3_ACTIVE = True                                    # False: T3 is logged but acts like T4
TIERS = (0, 1, 2, 3, 4)


# ---------------------------------------------------------------- verbatim from Phase-3 eval cell 5
def clean_and_prep_spectrum(mzs, ints, prec_mz=None, mask_precursor=True, int_floor=0.01, max_peaks=64, power=0.5):
    """
    Cleans raw peaks and computes probability distribution weights and Shannon entropy.
    Parameters:
      - prec_mz: Precursor m/z for precursor masking (|mz - prec_mz| > 1.5 Da).
      - int_floor: Relative threshold (drops peaks < int_floor * max_peak).
      - max_peaks: Keeps top N most intense peaks.
      - power: Intensity exponent (0.5 = square root).
    Returns:
      (cleaned_mzs, weights, entropy, is_empty)
    """
    if len(mzs) == 0:
        return np.array([]), np.array([]), 0.0, True

    mzs = np.asarray(mzs, dtype=np.float64)
    ints = np.asarray(ints, dtype=np.float64)

    if mask_precursor and prec_mz is not None:
        mask = np.abs(mzs - prec_mz) > 1.5
        mzs = mzs[mask]
        ints = ints[mask]
        if len(mzs) == 0:
            return np.array([]), np.array([]), 0.0, True

    max_i = np.max(ints)
    if max_i <= 0:
        return np.array([]), np.array([]), 0.0, True

    floor_mask = ints >= (int_floor * max_i)
    mzs = mzs[floor_mask]
    ints = ints[floor_mask]
    if len(mzs) == 0:
        return np.array([]), np.array([]), 0.0, True

    if len(mzs) > max_peaks:
        top_idx = np.argsort(ints)[-max_peaks:]
        mzs = mzs[top_idx]
        ints = ints[top_idx]

    order = np.argsort(mzs)
    mzs = mzs[order]
    ints = ints[order]

    w = np.power(ints, power)
    w_sum = np.sum(w)
    if w_sum <= 0:
        return np.array([]), np.array([]), 0.0, True
    w = w / w_sum

    entropy = -float(np.sum(w * np.log(w + 1e-15)))
    return mzs, w, entropy, False


def compare_spectra(prep1, prep2, prec_mz, mz_tol_ppm=15.0):
    """
    Fast comparison between two pre-cleaned spectra.
    Returns:
      entropy_sim: float [0, 1]
      cos_sim: float [0, 1]
      n_matched: int
      has_high_mz: bool
      max_matched_mz: float
    """
    m1, w1, s1, empty1 = prep1
    m2, w2, s2, empty2 = prep2

    if empty1 or empty2:
        return 0.0, 0.0, 0, False, 0.0

    half_prec = 0.5 * prec_mz if prec_mz is not None else 0.0
    j_start = 0
    n2 = len(m2)

    matched_j_set = set()
    matches = [] # (i, j, mz1, w1, w2)
    dot = 0.0
    matched_mzs = []

    norm1 = np.linalg.norm(w1)
    norm2 = np.linalg.norm(w2)

    for i, (mz_val, val1) in enumerate(zip(m1, w1)):
        tol = mz_val * mz_tol_ppm * 1e-6
        best_diff = tol
        best_j = -1
        for j in range(j_start, n2):
            diff = abs(mz_val - m2[j])
            if m2[j] < mz_val - tol:
                j_start = j
                continue
            if m2[j] > mz_val + tol:
                break
            if diff <= best_diff:
                best_diff = diff
                best_j = j

        if best_j != -1 and best_j not in matched_j_set:
            matched_j_set.add(best_j)
            val2 = w2[best_j]
            matches.append((i, best_j, mz_val, val1, val2))
            matched_mzs.append(mz_val)
            dot += val1 * val2

    n_matched = len(matches)
    has_high_mz = any(m > half_prec for m in matched_mzs)
    max_matched_mz = max(matched_mzs) if matched_mzs else 0.0
    cos_sim = float(dot / (norm1 * norm2)) if (norm1 > 0 and norm2 > 0) else 0.0

    # Mixture Spectrum M = (P + Q) / 2
    matched_i_set = {m[0] for m in matches}
    mixture_weights = []

    for _, _, _, v1, v2 in matches:
        mixture_weights.append((v1 + v2) * 0.5)

    for i, v1 in enumerate(w1):
        if i not in matched_i_set:
            mixture_weights.append(v1 * 0.5)

    for j, v2 in enumerate(w2):
        if j not in matched_j_set:
            mixture_weights.append(v2 * 0.5)

    mixture_weights = np.asarray(mixture_weights, dtype=np.float64)
    s_mix = -float(np.sum(mixture_weights * np.log(mixture_weights + 1e-15)))

    # Entropy distance D = 2 * S(M) - S(P) - S(Q)
    entropy_dist = max(0.0, 2.0 * s_mix - s1 - s2)
    entropy_sim = float(max(0.0, min(1.0, 1.0 - (entropy_dist / np.log(4.0)))))

    return entropy_sim, cos_sim, n_matched, has_high_mz, max_matched_mz
# ---------------------------------------------------------------- end verbatim


def _empty_decision(n_q=0, n_ref=0, secs=0.0):
    return dict(tier=4, S=0.0, rival=0.0, support=0, n_q=n_q, n_ref=n_ref, sid=-1, smiles=None,
                key=None, margin_demoted=False, secs=round(secs, 4))


def decide_tier(S, rival, support, n_q):
    """Tier of the best structure, and whether the rival-margin rule demoted it."""
    if S >= TAU1:
        tier = 1
    elif S >= TAU2:
        tier = 2
    elif S >= TAU3 and support >= min(2, max(n_q, 1)):
        tier = 3
    else:
        tier = 4
    demoted = False
    if tier < 4 and S - rival < MARGIN:
        tier, demoted = (3 if tier in (1, 2) else 4), True
    return tier, demoted


def evaluate_molecule(L, spectra, target, excl_lib_ix, adduct_index, identity_of_sid):
    """Gate decision for one molecule.

    spectra: Cell-18 dicts (mz, it, prec, mode, adduct). target: neutral mass (median nm).
    excl_lib_ix: L.lib codes to drop from the window. adduct_index: casmi.chem.adduct_index.
    identity_of_sid: sid -> metric identity (rival structures are counted per identity, not per sid).
    """
    t0 = time.time()
    centre = float(target) * (1.0 - PPM_CENTER * 1e-6)
    excl = np.asarray(list(excl_lib_ix), dtype=np.int64)
    best = {}        # sid -> best filtered entropy over (query spectrum, reference spectrum)
    support = {}     # sid -> set of query-spectrum indices with a filtered pair >= TAU3
    n_q = n_ref = 0
    for qi, sp in enumerate(spectra):
        prec = float(sp['prec'])
        q = clean_and_prep_spectrum(sp['mz'], sp['it'], prec_mz=prec, mask_precursor=True)
        if q[3]:
            continue
        n_q += 1
        cand = np.asarray(L.window(centre, PPM_WIN, mode=int(sp['mode'])), dtype=np.int64)
        if len(cand) == 0:
            continue
        keep = (L.sid[cand] >= 0) & (L.adduct_ix[cand] == adduct_index(sp['adduct']))
        if len(excl):
            keep &= ~np.isin(L.lib[cand], excl)
        cand = cand[keep]
        n_ref += len(cand)
        for c in cand:
            a, b = int(L.off[c]), int(L.off[c + 1])
            r = clean_and_prep_spectrum(np.asarray(L.mz[a:b], np.float64), np.asarray(L.it[a:b], np.float64),
                                        prec_mz=prec, mask_precursor=True)
            ent, _cos, n_m, high, _mx = compare_spectra(q, r, prec)
            if n_m < N_MIN or not high:
                continue
            sid = int(L.sid[c])
            if ent > best.get(sid, -1.0):
                best[sid] = ent
            if ent >= TAU3:
                support.setdefault(sid, set()).add(qi)
    if not best:
        return _empty_decision(n_q, n_ref, time.time() - t0)
    by_id = {}       # identity -> (S, sid)
    for sid, s in best.items():
        ident = identity_of_sid(sid)
        if ident is None:
            ident = ('sid', sid)
        if s > by_id.get(ident, (-1.0, -1))[0]:
            by_id[ident] = (s, sid)
    ranked = sorted(by_id.items(), key=lambda kv: -kv[1][0])
    ident, (S, sid) = ranked[0]
    rival = ranked[1][1][0] if len(ranked) > 1 else 0.0
    sup = len(support.get(sid, ()))
    tier, demoted = decide_tier(S, rival, sup, n_q)
    return dict(tier=tier, S=float(S), rival=float(rival), support=int(sup), n_q=int(n_q), n_ref=int(n_ref),
                sid=int(sid), smiles=str(L.struct_smiles[sid]), key=None if isinstance(ident, tuple) else str(ident),
                margin_demoted=bool(demoted), secs=round(time.time() - t0, 4))


def molecule_spectra(sub):
    """Cell-18 spectrum dicts from one molecule's test rows (same fields the engine receives)."""
    out = []
    for r in sub.itertuples():
        out.append(dict(mz=np.asarray(r.ms2_mzs, np.float64), it=np.asarray(r.ms2_normalized_intensities, np.float64),
                        mode=1 if r.ionization_mode == 'positive' else -1, adduct=r.adduct,
                        prec=float(r.precursor_mz)))
    return out


def run_all(L, mols, libs, adduct_index, identity_of_sid, budget_s=900.0, log_every=50):
    """{str(mid): decision} for every molecule, plus a JSON-safe summary.

    Molecules left after the time budget get no entry, so they keep legacy behaviour.
    """
    excl = [libs.index(x) for x in EXCL_LIBS if x in libs]
    assert len(excl) == len(EXCL_LIBS), f'library codes missing: {EXCL_LIBS} vs {libs}'
    t0 = time.time()
    gates, timed_out = {}, 0
    for gi, (mid, sub) in enumerate(mols):
        if time.time() - t0 > budget_s:
            timed_out = len(mols) - gi
            break
        nms = sub.nm.values[np.isfinite(sub.nm.values)]
        if not len(nms):
            gates[str(mid)] = _empty_decision()
            continue
        gates[str(mid)] = evaluate_molecule(L, molecule_spectra(sub), float(np.median(nms)), excl,
                                            adduct_index, identity_of_sid)
        if log_every and ((gi + 1) % log_every == 0):
            print(f'  smart gate {gi+1}/{len(mols)} {time.time()-t0:.0f}s', flush=True)
    summary = dict(health='ok' if not timed_out else f'partial: {timed_out} molecules over budget',
                   secs=round(time.time() - t0, 1), n=len(gates), timed_out=int(timed_out),
                   tiers={str(t): sum(d['tier'] == t for d in gates.values()) for t in TIERS[1:]},
                   margin_demoted=sum(bool(d['margin_demoted']) for d in gates.values()),
                   excl_lib_ix=excl, taus=[TAU1, TAU2, TAU3], n_min=N_MIN, margin=MARGIN)
    return gates, summary


def locked(dec):
    """T1/T2: no PubChem slots, no PubChem simulator candidates, rank-1 pin."""
    return bool(dec) and dec.get('tier') in (1, 2)


def merge_decision(smis, keys, lib_max, p, dec, *, LIB_TAU, REL_TH, SLOTS_AGG, SLOTS_GENTLE, S_TAU, POP_TAU,
                   merge, promote):
    """Cell-22 merge for one molecule -> (final list, stats labels).

    T4 (and dec=None) reproduces the original Cell-22 loop exactly; lib_max >= LIB_TAU stays untouched
    for every tier, so no molecule the original leaves untouched ever receives PubChem slots.
    """
    tier = dec.get('tier', 4) if dec else 4
    if tier == 3 and not T3_ACTIVE:
        tier = 4
    if p is None or not p['pc']:
        return smis, ['no_pc']
    if tier in (1, 2):
        return smis, ['untouched', f't{tier}']
    if lib_max >= LIB_TAU:
        return smis, ['untouched', 't0']
    rel = (p['fz_top'] if p.get('fz_top') is not None else p['pc_fz'][0]) - p['best_pool_fz'] if np.isfinite(p['best_pool_fz']) else 1e9
    if tier == 3:   # gentle late slots, never promoted over a filtered library match
        return merge(smis, keys, p['pc'], p['pc_keys'], SLOTS_T3), ['t3']
    slots = SLOTS_AGG if rel > REL_TH else SLOTS_GENTLE
    labels = ['aggressive' if rel > REL_TH else 'gentle']
    S = p.get('S'); pop_top = float(p.get('top_pop') or 0.0)
    if (S is not None and float(S) > S_TAU and pop_top >= POP_TAU) or (rel > 250.0):
        labels.append('promoted')
        return promote(smis, keys, p['pc'], p['pc_keys'], slots), labels
    return merge(smis, keys, p['pc'], p['pc_keys'], slots), labels


LOCK_EVENTS = ('pinned_moved', 'top2_moved', 'inserted', 'kept', 'identity_fail')


def apply_lock(result, dec, identity, safe_extend, limit=25):
    """Final-overlay lock on the TREATMENT list only -> (list, event or None).

    T1/T2: library structure at rank 1. T3: library structure at rank <= 2. Absent -> inserted there.
    Matching uses the overlay's own metric identity, never a stored key.
    """
    if not dec or dec.get('tier') not in ((1, 2, 3) if T3_ACTIVE else (1, 2)) or not dec.get('smiles'):
        return list(result), None
    lib_id = identity(dec['smiles'])
    if lib_id is None:
        return list(result), 'identity_fail'
    target = 0 if dec['tier'] in (1, 2) else 1
    pos = next((i for i, s in enumerate(result) if identity(s) == lib_id), None)
    rest = [s for i, s in enumerate(result) if i != pos]
    if pos is None:
        event, hit = 'inserted', dec['smiles']
    elif pos > target:
        event, hit = ('pinned_moved' if target == 0 else 'top2_moved'), result[pos]
    else:
        return list(result), 'kept'
    t = min(target, len(rest))
    return safe_extend(rest[:t] + [hit] + rest[t:], [], limit=limit), event
