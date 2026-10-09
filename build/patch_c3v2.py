"""Patch the 0.415 notebook with the gated Class-3 mutator v2 (specs/track4_priors_probes/c3v2_plan.md).

Source: casmi26-v29-fusion-pop-glmh-0-415.ipynb (same code as 0.411).  Output: casmi26-c3v2-<tag>.ipynb.
Always patches the untouched 0.415 source, never in place.  Feeding it an already patched notebook
fails loudly: the anchors it edits are consumed, so nothing can be patched twice.

What changes, cell by cell (everything sits inside a try/except; any failure = 0.415 behaviour):
  config   ICE_BUDGET 5400 -> --ice-budget (default 7200) for the generated candidates' ICE scoring
  engine   export `eng_lists.json.c3.json`: per molecule the exact-window library similarity and the
           top-20 mass-shifted analogs [smiles, exact mass, similarity] -- the anchors (I1)
  PubChem  export `pc_z.npz`, the FPNet logits z the channel already holds (the f.z screen, I0)
  NEW      mutator cell after PubChem: mut_runner.py in MUT_V2 mode (anchored, signed, exact-mass
           delta, tau-pruned, stage-1 gate) -> MUT {mid: smiles, keys, formula '|gen', fz, tier, sig}
  ICE      MUT guard; generated candidates join the ICE/GLACIER input under '<formula>|gen', so they
           are scored but never share a same-formula group with a retrieved candidate;
           GL_BUDGET 4000 -> 5200
  fusion   after the 0.415 fused rerank is final: stage-2 gate (real lib_max, ICE/GL competition,
           engine agreement), run-wide cap, then a splice of <= 3 generated structures into
           sure -> [1,2,3] / likely -> [2,3,4].  The fused rerank itself is untouched.
           --no-splice (probe C1-null): everything above still runs and prints its counters, but the
           merge is skipped, so the score measures the plumbing alone (budgets, ICE/GL union).

usage: python build/patch_c3v2.py [--tau 8] [--beta 0] [--no-rank1] [--sure-frac 0.03]
                                  [--fire-frac 0.12] [--ice-budget 7200] [--no-splice]
                                  [--src nb] [--out nb]
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from patch_v4n_mutator import code, esc, md, must_replace, set_src, src_of  # noqa: E402

SRC = os.path.join(ROOT, "casmi26-v29-fusion-pop-glmh-0-415.ipynb")
MARK = "# === C3V2 MUTATOR CHANNEL (track4 c3v2) ==="
MD_MARK = "## Class-3 mutator v2"
SPLICE_MARK = "    # === C3V2 SPLICE ==="

MUT_CELL = MARK + '''
# Anchored, signed Delta-M mutation for suspected Class-3 molecules, in its own CPU process (the
# casmi26-v3-models snapshot that produced z).  Any failure leaves MUT = {} -> 0.415 behaviour.
MUT_ON, MUT_K, MUT_N1, MUT_DEADLINE = True, 25, 2000, 45
C3_SPLICE_ON = {SPLICE_ON}   # False = attribution probe C1-null: gate and counters run, lists untouched
C3V2_ENV = {C3ENV}
for _k, _v in C3V2_ENV.items():
    os.environ[_k] = str(_v)               # the fusion cell imports c3_gate in-process; same thresholds
MUT_DIR = '/kaggle/working/mut'; os.makedirs(MUT_DIR, exist_ok=True)
MUT_FILES = {}
MUT_FILES['delta_m_mutator.py'] = {DM_SRC}
MUT_FILES['mut_runner.py'] = {MR_SRC}
MUT_FILES['c3_gate.py'] = {C3_SRC}
for _n, _s in MUT_FILES.items():
    open(os.path.join(MUT_DIR, _n), 'w').write(_s)

MUT = {}
_engf = '/kaggle/working/eng_lists.json'
_c3f = _engf + '.c3.json'
if MUT_ON and PC and os.path.exists('/kaggle/working/pc_z.npz') and os.path.exists(_c3f):
    try:
        _t0 = time.time()
        _env = dict(os.environ, MUT_V2='1', MUT_K=str(MUT_K), MUT_N1=str(MUT_N1),
                    MUT_DEADLINE=str(MUT_DEADLINE), MUT_C3=_c3f)
        if ENG and os.path.exists(_engf):
            _env['MUT_ENG'] = _engf          # engine keys: generated structures must be new to it too
        _cmd = [sys.executable, os.path.join(MUT_DIR, 'mut_runner.py'), V3CODE, POOL_DIR,
                '/kaggle/working/pc_lists.json', '/kaggle/working/pc_z.npz',
                '/kaggle/working/mut_lists.json', '4']
        rr = subprocess.run(_cmd, capture_output=True, text=True, timeout=2 * 3600, env=_env)
        print(rr.stdout[-4000:]); print(rr.stderr[-2000:])
        MUT = json.load(open('/kaggle/working/mut_lists.json'))
    except Exception as e:
        print('C3V2 MUTATOR FAILED -> 0.415 behaviour kept:', repr(e))
else:
    print('C3V2 mutator skipped (MUT_ON=%s, PC=%d, pc_z=%s, c3=%s)'
          % (MUT_ON, len(PC), os.path.exists('/kaggle/working/pc_z.npz'), os.path.exists(_c3f)))

# ---- sanity: shape, novelty, isolation tag, tiers.  Never assert in a scoring run.
_bad = []
for _m, _v in MUT.items():
    if not (len(_v['smiles']) == len(_v['keys']) == len(_v['formula']) == len(_v['fz'])):
        _bad.append(f'{_m}: ragged lists')
    if len(set(_v['keys'])) != len(_v['keys']):
        _bad.append(f'{_m}: duplicate keys')
    if len(_v['smiles']) > MUT_K:
        _bad.append(f'{_m}: {len(_v["smiles"])} > MUT_K')
    if not all(str(f).endswith('|gen') for f in _v['formula']):
        _bad.append(f'{_m}: formula without |gen (would mix with retrieved isomers)')
    if _v.get('tier') not in ('sure', 'likely'):
        _bad.append(f'{_m}: tier {_v.get("tier")!r}')
    if set((PC.get(_m) or {}).get('pc_keys', [])) & set(_v['keys']):
        _bad.append(f'{_m}: overlaps its own PubChem list')
    if set((ENG.get(_m) or {}).get('keys', [])) & set(_v['keys']):
        _bad.append(f'{_m}: overlaps its own engine list')
_tiers = {}
for _v in MUT.values():
    _tiers[_v.get('tier')] = _tiers.get(_v.get('tier'), 0) + 1
print(f'C3V2 STAGE 1: {len(MUT)} molecules fired {_tiers} | '
      f'{sum(len(v["smiles"]) for v in MUT.values())} candidates | sanity {"OK" if not _bad else _bad[:5]}',
      flush=True)
if _bad:
    print('C3V2 SANITY FAILED -> channel disabled, 0.415 behaviour kept:', _bad[:10])
    MUT = {}
print(f'c3v2 mutator {time.time()-T0:.0f}s')
'''

ICE_BLOCK = '''    if MUT:                              # ADDED (C3V2): generated candidates join the ICE/GL input under
        n_mut_add = 0                    # '<formula>|gen', so they are scored but form their own groups.
        for m in mids:
            u = MUT.get(str(m))
            if not u or not u.get('smiles'):
                continue
            _ms, _mf = list(u['smiles'][:25]), list(u['formula'][:25])
            _before = len(cands.get(m, []))
            cands[m] = list(dict.fromkeys(list(cands.get(m, [])) + ice_fuse.ice_candidates(_ms, _mf, len(_ms))))
            n_mut_add += len(cands[m]) - _before
        print('C3V2 ICE_UNION_MUT: generated candidates added to ICE input:', n_mut_add, flush=True)
'''

SPLICE = SPLICE_MARK + '''
    # Stage 2 + run-wide cap + splice.  Runs after the 0.415 fused rerank is final, so generated
    # structures never enter a same-formula group with a retrieved candidate (Track-1 friendly fire).
    if MUT:
        try:
            if MUT_DIR not in sys.path:
                sys.path.insert(0, MUT_DIR)
            import c3_gate as C3G
            C3_SLOTS = {'sure': [1, 2, 3], 'likely': [2, 3, 4]}
            _v4top = {}
            for _m, _s in zip(v4.molecule_id, v4.smiles):
                _f = [x for x in str(_s).split(';') if x and x != 'CCO']
                _v4top[str(_m)] = (chem.score_key(_f[0]) or _f[0]) if _f else None
            def _look(d, s_, k_):
                if not d:
                    return None
                v = d.get(k_, d.get(s_))
                return None if v is None else float(v)
            _all_ice = [float(v) for d in (ICE_SCORES.values() if isinstance(ICE_SCORES, dict) else [])
                        if isinstance(d, dict) for v in d.values() if v is not None]
            _ice_unit = bool(_all_ice) and min(_all_ice) >= -1e-6 and max(_all_ice) <= 1 + 1e-6
            def _z(a):
                a = np.asarray([np.nan if x is None else x for x in a], np.float64)
                ok = np.isfinite(a)
                if ok.sum() < 2:
                    return np.zeros(len(a))
                s_ = a[ok].std()
                return np.where(ok, (a - np.nanmean(a)) / (s_ + 1e-9), 0.0)
            _r2, _rows, _cand = collections.Counter(), [], {}
            for _i, (mid, s) in enumerate(out):
                u = MUT.get(str(mid))
                if not u or not u.get('smiles') or u.get('tier') not in C3_SLOTS:
                    continue
                vs = [x for x in s.split(';') if x and x != 'CCO']
                _ms, _mk = list(u['smiles']), list(u['keys'])
                _ice = ICE_SCORES.get(str(mid), {}) if isinstance(ICE_SCORES, dict) else {}
                _gl = gl_of(mid)
                _gi = [_look(_ice, a, b) for a, b in zip(_ms, _mk)]
                _gg = [_look(_gl, a, b) for a, b in zip(_ms, _mk)]
                _o = np.argsort(-(_z(u['fz']) + _z(_gi) + _z(_gg)), kind='stable')
                _ms = [_ms[j] for j in _o]; _mk = [_mk[j] for j in _o]
                _gi = [_gi[j] for j in _o]; _gg = [_gg[j] for j in _o]
                _rk = (chem.score_key(vs[0]) or vs[0]) if vs else None
                _ri = _look(_ice, vs[0], _rk) if vs else None
                _rg = _look(_gl, vs[0], _rk) if vs else None
                _b = BASE.get(mid, BASE.get(str(mid)))
                _lib = float(_b[2]) if _b is not None else None
                _ek = (ENG.get(str(mid)) or {}).get('keys', [])[:1]
                _agree = bool(_ek) and _v4top.get(str(mid)) is not None and _ek[0] == _v4top[str(mid)]
                _t2 = C3G.classify_v2(u['tier'], _lib, _agree, _gi[0], _ri, _gg[0], _rg,
                                      reasons=_r2, ice_unit=_ice_unit)
                _sig = u.get('sig') or {}
                _conf = (float(_sig.get('ana_max') or 0) - float(_sig.get('lib_exact') or 0)
                         + ((_gi[0] - _ri) if (_gi[0] is not None and _ri is not None) else 0.0))
                _rows.append((str(mid), _t2, _conf)); _cand[str(mid)] = (_i, _ms, _mk)
            _fin, _cs = C3G.cap_fires(_rows, len(out))
            _on = globals().get('C3_SPLICE_ON', True)
            n_splice, n_t1, _ntier = 0, 0, {}
            for _m, _t in _fin.items():
                if not _t:
                    continue
                _i, _ms, _mk = _cand[_m]
                mid, s = out[_i]
                vs = [x for x in s.split(';') if x and x != 'CCO']
                vs2 = merge(vs, [chem.score_key(x) or x for x in vs], _ms[:3], _mk[:3], C3_SLOTS[_t])
                n_t1 += int(bool(vs) and vs2[:1] != vs[:1])
                if _on:
                    out[_i] = (mid, ';'.join(vs2[:25]) if vs2 else 'CCO')
                n_splice += 1; _ntier[_t] = _ntier.get(_t, 0) + 1
            if not _on:
                print(f'C3V2 SPLICE DISABLED (--no-splice): would splice {n_splice} molecules {_ntier}, '
                      f'top-1 would change in {n_t1}; lists left as 0.415', flush=True)
            print(f'C3V2 STAGE 2: {dict(_r2)} | ice scale in [0,1]: {_ice_unit}', flush=True)
            print(f'C3V2 CAP: {dict(_cs)}', flush=True)
            print(f'C3V2 SPLICE: {n_splice} molecules {_ntier} | top-1 changed by splice {n_t1} | '
                  f'max |dMRR| bound {(_ntier.get("sure", 0) + 0.5 * _ntier.get("likely", 0)) / max(1, len(out)):.4f}',
                  flush=True)
        except Exception as ex:
            print('C3V2 SPLICE FAILED -> 0.415 lists kept:', repr(ex))
'''


def main(argv):
    opts = dict(tau="8", beta="0", sure_frac="0.03", fire_frac="0.12", ice_budget="7200",
                src=SRC, out=None)
    rank1, splice = True, True
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--no-rank1":
            rank1 = False
            i += 1
            continue
        if a == "--no-splice":
            splice = False
            i += 1
            continue
        k = a.lstrip("-").replace("-", "_")
        if k not in opts:
            raise SystemExit("unknown flag " + a)
        opts[k] = argv[i + 1]
        i += 2
    tag = "tau%s" % opts["tau"] + ("_b%s" % opts["beta"] if float(opts["beta"]) else "") + \
          ("" if rank1 else "_norank1") + ("" if splice else "_nosplice")
    dst = opts["out"] or os.path.join(ROOT, "casmi26-c3v2-%s.ipynb" % tag)

    nb = json.load(open(opts["src"], encoding="utf-8"))
    cells = [c for c in nb["cells"] if MARK not in src_of(c) and MD_MARK not in src_of(c)]
    nb["cells"] = cells

    def find(pred, what):
        hits = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code" and pred(src_of(c))]
        if len(hits) != 1:
            raise SystemExit(f"PATCH FAILED: {what} matched {len(hits)} cells")
        return hits[0]

    i_cfg = find(lambda s: "TOPN, ICE_LAM, ICE_BUDGET, ICE_PC" in s, "config cell")
    i_eng = find(lambda s: "eng_runner.py" in s, "engine cell")
    i_pc = find(lambda s: "pc_runner.py" in s and "RUNNER" in s, "PubChem cell")
    i_ice = find(lambda s: "ICE_UNION" in s and "import gc" in s, "ICEBERG cell")
    i_fuse = find(lambda s: "def fuse2" in s, "fusion cell")

    # ---- config: ICE budget for the generated candidates
    t = src_of(nb["cells"][i_cfg])
    t = must_replace(t, "TOPN, ICE_LAM, ICE_BUDGET, ICE_PC = 60, 1.0, 5400, True",
                     "TOPN, ICE_LAM, ICE_BUDGET, ICE_PC = 60, 1.0, %d, True   # C3V2: +budget for generated candidates"
                     % int(float(opts["ice_budget"])), "ICE_BUDGET")
    set_src(nb["cells"][i_cfg], t)

    # ---- engine: export the anchors (analogs with exact masses) and the exact-window library match
    t = src_of(nb["cells"][i_eng])
    blk = ("    c3 = {}\n"
           "    for r in recs:\n"
           "        _a = r.get('ana') or []\n"
           "        _l = r.get('lib')\n"
           "        c3[str(r['mid'])] = dict(\n"
           "            target=float(r.get('target') or 0.0),\n"
           "            lib_exact=(float(np.max(_l)) if _l is not None and len(_l) else 0.0),\n"
           "            ana=[[str(E.POOL['smiles'][int(_p)]), float(E.POOL['mass'][int(_p)]),\n"
           "                  float(_s)] for _p, _s in _a[:20]])\n"
           "    json.dump(c3, open(out + '.c3.json', 'w'))\n"
           "    print('c3 diagnostics', len(c3), flush=True)\n")
    t = must_replace(t, esc("    json.dump(res, open(out, 'w'))\n"),
                     esc(blk) + esc("    json.dump(res, open(out, 'w'))\n"), "c3 diagnostics export")
    set_src(nb["cells"][i_eng], t)

    # ---- PubChem: export z
    t = src_of(nb["cells"][i_pc])
    anchor = "    json.dump(res, open(OUT, \\'w\\'))\\n"
    add = ("    np.savez_compressed(os.path.join(os.path.dirname(OUT), \\'pc_z.npz\\'),\\n"
           "                        mids=np.array([t[0] for t in tasks], dtype=object),\\n"
           "                        z=np.stack([t[2] for t in tasks]).astype(np.float32))\\n")
    t = must_replace(t, anchor, add + anchor, "pc_z export")
    set_src(nb["cells"][i_pc], t)

    # ---- ICE: guard, generated candidates in the union, GLACIER budget
    t = src_of(nb["cells"][i_ice])
    t = must_replace(t, "import gc\n", "import gc\nMUT = globals().get('MUT') or {}\n", "MUT guard")
    t = must_replace(t, "    if ICE_PC:", ICE_BLOCK + "    if ICE_PC:", "ICE_UNION_MUT")
    t = must_replace(t, "dict(molecules=0, changed_vs_ice_top25=0, changed_vs_ice_top1=0), 1.0, 4000",
                     "dict(molecules=0, changed_vs_ice_top25=0, changed_vs_ice_top1=0), 1.0, 5200",
                     "GL_BUDGET")
    set_src(nb["cells"][i_ice], t)

    # ---- fusion: run when either list exists; splice after the fused lists are final
    t = src_of(nb["cells"][i_fuse])
    t = must_replace(t, "if ENG:\n    v4 = pd.read_csv('submission.csv')",
                     "import collections\nMUT = globals().get('MUT') or {}\n"
                     "if ENG or MUT:\n    v4 = pd.read_csv('submission.csv')", "fusion guard")
    t = must_replace(t, "    fs = pd.DataFrame(out, columns=['molecule_id', 'smiles'])",
                     SPLICE + "    fs = pd.DataFrame(out, columns=['molecule_id', 'smiles'])", "splice")
    set_src(nb["cells"][i_fuse], t)

    # ---- the mutator cell, right after the PubChem cell (PC, pc_z.npz and the engine export exist)
    env = {"C3_MAX_STEPS_V2": 2, "C3_SURE_FRAC": opts["sure_frac"], "C3_FIRE_FRAC": opts["fire_frac"],
           "C3_ALLOW_RANK1": "1" if rank1 else "0", "MUT_TAU": opts["tau"], "MUT_BETA": opts["beta"]}
    dm = open(os.path.join(HERE, "delta_m_mutator.py"), encoding="utf-8").read()
    mr = open(os.path.join(HERE, "mut_runner.py"), encoding="utf-8").read()
    cg = open(os.path.join(HERE, "c3_gate.py"), encoding="utf-8").read()
    body = (MUT_CELL.replace("{C3ENV}", repr(env)).replace("{SPLICE_ON}", repr(splice)).replace("{DM_SRC}", repr(dm))
            .replace("{MR_SRC}", repr(mr)).replace("{C3_SRC}", repr(cg)))
    nb["cells"].insert(i_pc + 1, code(body))
    note = (
        MD_MARK + " (track4 c3v2)\n"
        "Fires only on molecules that look Class 3, then splices at most 3 generated structures into "
        "reserved ranks.\n\n"
        "- **Anchors**: engine-A mass-shifted spectral analogs whose exact mass delta to the query matches "
        "<= 2 signed moves (add or remove: hydroxyl, methyl, sugars, acyl groups, ...).\n"
        "- **Generation**: the whole analog is edited, not its Murcko scaffold; the frontier is ranked "
        "by f.z against the query's predicted fingerprint and intermediates scoring more than tau below "
        "the anchor are pruned.\n"
        "- **Stage 1**: the best edit must beat its anchor, the pool and (for sure) the PubChem-only top "
        "on f.z -- within-molecule comparisons, no absolute f.z cut.\n"
        "- **Stage 2**: real library match, ICEBERG/GLACIER competition against the retrieved top-1, "
        "engine agreement; a run-wide cap (sure <= %s, fire <= %s, off above 25%%).\n"
        "- **Splice**: sure -> ranks [1,2,3]%s, likely -> [2,3,4], after the fused rerank, so generated "
        "structures never share a same-formula group with retrieved ones."
        % (opts["sure_frac"], opts["fire_frac"], "" if rank1 else " (disabled: likely slots)"))
    nb["cells"].insert(i_pc + 1, md(note))

    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
        c.get("metadata", {}).pop("papermill", None)
    nb.get("metadata", {}).pop("papermill", None)
    json.dump(nb, open(dst, "w", encoding="utf-8"), indent=1)
    print("wrote %s (%d cells; tau=%s beta=%s rank1=%s splice=%s)" % (dst, len(nb["cells"]), opts["tau"],
                                                                     opts["beta"], rank1, splice))
    return dst


if __name__ == "__main__":
    main(sys.argv[1:])
