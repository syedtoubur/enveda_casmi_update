"""Track 4c probe generator: a spectral NP class vote on the zenith fused list.

Plan: specs/track4_priors_probes/np_class_sig_high_influence.md (register T4C-* in gate_report.md).
Never hand-edit the generated .ipynb -- change this script and regenerate. The source notebook is read, never
written. Outputs are cleared so every variant uploads cleanly.

--classvote MU   T4C  after fuse2 and before the GLACIER rerank, re-sort the fused list by z(RRF) + MU*t:
                      * voters: train-library spectra at the query neutral mass or one NP biosynthetic delta away
                        (scratch_pkg analog.NP_DELTAS, each +-10 ppm, same polarity), scored by the v4b library's
                        shift-aware hybrid entropy (casmi.spectra.search_shift); top 10 structures with sim >= 0.5
                      * labels: NP-table superclass by scorer_key, else nearest labelled NP-table structure
                        (Morgan r2 2048, Tanimoto >= 0.7), else none (abstains / t = 0)
                      * t = +share(label) if the candidate's superclass got votes, -share(winner) if it is labelled
                        with an unvoted class, 0 if unlabelled
                      * fires only if lib_max < LIB_TAU, >= 2 labelled voters and winner share >= 0.5
                      Commit runs also print a dry-reach line (lib_max gate ignored, nothing applied).
                      Needs the NP knowledge table (natural_products.parquet) attached.

usage: python build/patch_zenith.py --classvote 0.5 [--src in.ipynb] [--out out.ipynb]
"""
from __future__ import annotations

import argparse
import ast
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "enveda-casmi-2026-v18-sovereign-zenith.ipynb")
ANALOG = os.path.join(ROOT, "scratch_pkg", "casmi_scratch", "analog.py")


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


def np_deltas(path=ANALOG):
    """NP_DELTAS of the specs_scratch analog module (one shared definition; the tests check for drift)."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "NP_DELTAS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise SystemExit("PATCH FAILED: NP_DELTAS not found in " + path)


# ---------------------------------------------------------------- pure vote / term (also exec'd by the tests)
VOTE_FUNCS = r'''
def _cv_parts(lab):
    return [p.strip() for p in str(lab).replace(';', '|').split('|') if p.strip()] if isinstance(lab, str) and lab else []


def cv_vote(hits, label_of, k=10, min_sim=0.5):
    """hits {voter: sim} -> ({superclass: share}, winner share, n labelled voters). Top k with sim >= min_sim vote
    their label weighted by sim; a multi-valued label splits its weight; unlabelled voters abstain."""
    top = sorted(((s, v) for v, s in hits.items() if s >= min_sim), key=lambda x: (-x[0], str(x[1])))[:k]
    w, n = {}, 0
    for s, v in top:
        parts = _cv_parts(label_of(v))
        if not parts:
            continue
        n += 1
        for p in parts:
            w[p] = w.get(p, 0.0) + s / len(parts)
    tot = sum(w.values())
    if tot <= 0:
        return {}, 0.0, 0
    share = {p: x / tot for p, x in w.items()}
    return share, max(share.values()), n


def cv_term(labels, share, win):
    """Signed agreement per candidate: +share of its best voted superclass, -win if labelled but unvoted, 0 if unlabelled."""
    out = []
    for lab in labels:
        parts = _cv_parts(lab)
        if not parts:
            out.append(0.0)
            continue
        got = [share[p] for p in parts if p in share]
        out.append(max(got) if got else -win)
    return out


def cv_apply(smis, keys, scs, term, mu):
    """Re-sort a fused list by z(scs) + mu*term (stable). Returns the new lists, the new scores and the order."""
    s = np.asarray(scs, np.float64)
    z = (s - s.mean()) / s.std() if len(s) > 1 and s.std() > 0 else np.zeros(len(s))
    f = z + mu * np.asarray(term, np.float64)
    o = [int(i) for i in np.argsort(-f, kind='stable')]
    return [smis[i] for i in o], [keys[i] for i in o], [float(f[i]) for i in o], o
'''

CV_BLOCK = r'''# === TRACK 4c (patch_zenith.py): spectral NP class vote on the fused list ===
CV_MU, CV_K, CV_MIN_SIM, CV_MIN_VOTERS, CV_MIN_WIN, CV_PPM, CV_BUDGET = @MU@, 10, 0.5, 2, 0.5, 10.0, 1800.0
CV_DELTAS = @DELTAS@
CV_STATS = dict(molecules=0, searched=0, with_hits=0, voted=0, gated_libmax=0, fired=0, top1_changed=0,
                top25_changed=0, labelled_voters=0, voters=0, knn_queries=0, failed=0, over_budget=0, secs=0.0)
CV_DRY = dict(voted=0, fired=0, top1_changed=0)
CV_VOTES, _CV_TERMS = {}, []
@VOTE_FUNCS@
try:
    _cv_t0 = time.time()
    from rdkit import Chem as _CC, DataStructs as _CDS, RDLogger as _CRL
    from rdkit.Chem import rdFingerprintGenerator as _CFG
    from casmi import spectra as _csp
    from casmi.library import SearchCfg as _CSC
    _CRL.DisableLog('rdApp.*')
    _CCFG = _CSC()
    _cgen = _CFG.GetMorganGenerator(radius=2, fpSize=2048)
    def _cfp(s):
        m = _CC.MolFromSmiles(s) if isinstance(s, str) else None
        return None if m is None else _cgen.GetFingerprint(m)
    _npt = pd.read_parquet(find('natural_products.parquet'), columns=['scorer_key', 'smiles', 'np_superclass'])
    _npt = _npt[_npt.np_superclass.notna()]
    _CLAB = dict(zip(_npt.scorer_key, _npt.np_superclass))
    _CBANK, _CBL = [], []
    for _s, _l in zip(_npt.smiles, _npt.np_superclass):
        _fp = _cfp(_s)
        if _fp is not None:
            _CBANK.append(_fp); _CBL.append(_l)
    _CCACHE = {}
    def cv_label(smi, key):
        if key and key in _CLAB:
            return _CLAB[key]
        ck = key or smi
        if ck not in _CCACHE:
            fp = _cfp(smi); lab = None
            if fp is not None and _CBANK:
                CV_STATS['knn_queries'] += 1
                sims = np.asarray(_CDS.BulkTanimotoSimilarity(fp, _CBANK)); j = int(sims.argmax())
                lab = _CBL[j] if sims[j] >= 0.7 else None
            _CCACHE[ck] = lab
        return _CCACHE[ck]
    def _cv_lib_label(sid):
        return cv_label(L.struct_smiles[sid], L.struct_key[sid])
    print('TRACK4c classvote: NP-table labels', len(_CLAB), '| kNN bank', len(_CBANK), '| MU', CV_MU, flush=True)
    _cv_mols = {str(m): s for m, s in te.groupby('molecule_id', sort=False)}
    _cv_ok_sid = L.sid >= 0
    def cv_hits(sub):
        """{library structure sid: best shift-aware similarity} over the molecule's spectra (delta-list windows)."""
        nms = sub.nm.values[np.isfinite(sub.nm.values)]
        if not len(nms):
            return {}
        target = float(np.median(nms)) / (1.0 + 1.45e-6)          # timsTOF +1.45 ppm precursor offset
        parts = [L.window(target, CV_PPM)]
        for d in CV_DELTAS.values():
            for t in (target - d, target + d):
                if t > 0:
                    parts.append(L.window(t, CV_PPM))
        cand_all = np.unique(np.concatenate(parts))
        cand_all = cand_all[_cv_ok_sid[cand_all]]
        best = {}
        for r in sub.itertuples():
            mode = 1 if r.ionization_mode == 'positive' else -1
            cand = cand_all[L.mode[cand_all] == mode]
            if not len(cand):
                continue
            qm, qp = L.prep(np.asarray(r.ms2_mzs, np.float64), np.asarray(r.ms2_normalized_intensities, np.float64), _CCFG)
            if not len(qm):
                continue
            shift = (target - L.smass[cand]).astype(np.float64)
            sc = _csp.search_shift(qm, qp, cand, shift, L.off, L.mz, L.it, _CCFG.tol, _CCFG.ppm_tol,
                                   _CCFG.floor, _CCFG.topk, _CCFG.power, _CCFG.ent_weight)
            for c, s in zip(cand, sc):
                sid = int(L.sid[c])
                if s > best.get(sid, -1.0):
                    best[sid] = float(s)
        return best
    def cv_molecule(mid):
        """Vote for one molecule once (cached): (share, winner share, n labelled voters)."""
        mid = str(mid)
        if mid not in CV_VOTES:
            CV_VOTES[mid] = ({}, 0.0, 0)
            if time.time() - _cv_t0 > CV_BUDGET:
                CV_STATS['over_budget'] += 1
            elif mid in _cv_mols:
                CV_STATS['searched'] += 1
                h = cv_hits(_cv_mols[mid])
                CV_STATS['with_hits'] += int(any(s >= CV_MIN_SIM for s in h.values()))
                v = cv_vote(h, _cv_lib_label, k=CV_K, min_sim=CV_MIN_SIM)
                CV_STATS['voters'] += min(CV_K, sum(s >= CV_MIN_SIM for s in h.values()))
                CV_STATS['labelled_voters'] += v[2]
                CV_STATS['voted'] += int(v[2] > 0)
                CV_VOTES[mid] = v
        return CV_VOTES[mid]
    def cv_fused(mid, fsm, fk, fsc, dry=False):
        """Fused list -> (fsm, fk, fsc), re-sorted by z(RRF) + CV_MU*t when the gate passes. dry: count, don't apply."""
        share, win, n = cv_molecule(mid)
        if n < CV_MIN_VOTERS or win < CV_MIN_WIN or len(fsm) < 2:
            return fsm, fk, fsc
        t = cv_term([cv_label(s, k) for s, k in zip(fsm, fk)], share, win)
        if not any(t):
            return fsm, fk, fsc
        s2, k2, f2, o = cv_apply(fsm, fk, fsc, t, CV_MU if CV_MU > 0 else 1.0)
        if dry:
            CV_DRY['fired'] += 1; CV_DRY['top1_changed'] += int(o[0] != 0)
            return fsm, fk, fsc
        if CV_MU <= 0:
            return fsm, fk, fsc
        CV_STATS['fired'] += 1; CV_STATS['top1_changed'] += int(o[0] != 0)
        CV_STATS['top25_changed'] += int(o[:25] != list(range(min(25, len(o)))))
        _CV_TERMS.extend(abs(x) for x in t)
        return s2, k2, f2
    def cv_hook(mid, fsm, fk, fsc):
        """Never raises: any failure keeps zenith's fused order."""
        try:
            CV_STATS['molecules'] += 1
            if not IS_RERUN:                                     # dry reach: the public file is all class 1
                cv_fused(mid, fsm, fk, fsc, dry=True)
                CV_DRY['voted'] += int(CV_VOTES.get(str(mid), ({}, 0.0, 0))[2] > 0)
            if BASE.get(mid, ([], [], 0.0))[2] >= LIB_TAU:
                CV_STATS['gated_libmax'] += 1
                return fsm, fk, fsc
            return cv_fused(mid, fsm, fk, fsc)
        except Exception as _e:
            CV_STATS['failed'] += 1
            if CV_STATS['failed'] <= 3:
                print('TRACK4c classvote failed', mid, repr(_e), flush=True)
            return fsm, fk, fsc
except Exception as _e:
    print('TRACK4c classvote FAILED -> no-op:', repr(_e), flush=True)
    def cv_hook(mid, fsm, fk, fsc):
        return fsm, fk, fsc
'''

CV_REPORT = r'''    fs.to_csv('submission.csv', index=False)
    try:
        CV_STATS['secs'] = round(time.time() - _cv_t0, 1)
    except NameError:
        pass
    _ct = np.asarray(_CV_TERMS or [0.0])
    print('TRACK4c classvote | MU', CV_MU, '|', CV_STATS, f'| |term| mean/p99 {_ct.mean():.3f}/{np.percentile(_ct, 99):.3f}', flush=True)
    print('TRACK4c dry-reach (lib_max gate ignored, not applied) |', CV_DRY, flush=True)
    try:
        _man = json.load(open('run_manifest.json'))
        _man.update(classvote=dict(mu=CV_MU, stats=CV_STATS, dry=CV_DRY))
        json.dump(_man, open('run_manifest.json', 'w'), indent=1)
    except Exception as _e:
        print('TRACK4c manifest update failed', repr(_e))
'''


def patch_fuse(t, a):
    deltas = np_deltas()
    block = (CV_BLOCK.replace("@MU@", repr(float(a.classvote))).replace("@DELTAS@", repr(deltas))
             .replace("@VOTE_FUNCS@", VOTE_FUNCS))
    t = must_replace(t, "\nif ENG:\n", "\n" + block + "\nif ENG:\n", "class-vote block")
    t = must_replace(t, "            fsm, fk, fsc = fuse2(vs, e['smiles'], e['keys'])\n",
                     "            fsm, fk, fsc = fuse2(vs, e['smiles'], e['keys'])\n"
                     "            fsm, fk, fsc = cv_hook(mid, fsm, fk, fsc)\n", "class-vote hook")
    t = must_replace(t, "    fs.to_csv('submission.csv', index=False)\n", CV_REPORT, "class-vote report")
    return t


def build(a):
    nb = json.load(open(a.src, encoding="utf-8"))
    i_fuse = find(nb, lambda s: "def fuse2" in s, "fusion cell")
    set_src(nb["cells"][i_fuse], patch_fuse(src_of(nb["cells"][i_fuse]), a))
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
        c.get("metadata", {}).pop("papermill", None)
    return nb


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--classvote", type=float, required=True)
    p.add_argument("--src", default=SRC)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    dst = a.out or os.path.join(ROOT, f"casmi26-t4c-classvote{a.classvote:g}.ipynb")
    nb = build(a)
    json.dump(nb, open(dst, "w", encoding="utf-8"), indent=1)
    print("wrote", os.path.abspath(dst), "| classvote mu", a.classvote)
    return dst


if __name__ == "__main__":
    main()
