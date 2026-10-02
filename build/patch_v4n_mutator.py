"""Patch the forked 0.399 notebook with the Delta-M mutator channel (specs_v4n_update M1/M3/M4).

Idempotent: re-running replaces the injected cells rather than stacking them. It edits a copy,
never the fork in place.

Two variants, differing only in where the mutator harvests its scaffolds:

  default  PubChem top-25 only            -> casmi26-v4n-mutator-submit.ipynb
  --bio    PubChem top-25 + engine-A top-40 -> casmi26-v4n-mutator-bio-submit.ipynb

Engine A's list comes from a different structure universe (COCONUT + ChEBI/LIPID MAPS + training
structures, i.e. BIO) ranked by spectral evidence (AFIX analog propagation, MetFrag-lite, FPNet)
rather than by fingerprint f.z, so it is an orthogonal scaffold source. Under --bio the kept list
grows from 25 to 40 so engine-derived products add to the PubChem-derived ones instead of
displacing them, and the ICEBERG union is capped at 25 to hold the forward-simulation load flat.

usage: python build/patch_v4n_mutator.py [--bio] [in.ipynb] [out.ipynb]
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "casmi26-v4n-engine-fusion-union-lb-0-399.ipynb")
DST = os.path.join(ROOT, "casmi26-v4n-mutator-submit.ipynb")
DST_BIO = os.path.join(ROOT, "casmi26-v4n-mutator-bio-submit.ipynb")
DST_C3 = os.path.join(ROOT, "casmi26-v4n-mutator-c3-submit.ipynb")

MARK = "# === MUTATOR CHANNEL (specs_v4n_update) ==="


def code(src, marker=None):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": (src if src.endswith("\n") else src + "\n").splitlines(keepends=True)}


def md(src):
    return {"cell_type": "markdown", "metadata": {},
            "source": (src if src.endswith("\n") else src + "\n").splitlines(keepends=True)}


def src_of(cell):
    return "".join(cell["source"])


def must_replace(text, old, new, what):
    if old not in text:
        raise SystemExit(f"PATCH FAILED: anchor not found for {what}:\n  {old[:120]}")
    if text.count(old) != 1:
        raise SystemExit(f"PATCH FAILED: anchor for {what} occurs {text.count(old)} times")
    return text.replace(old, new)


def esc(block):
    """Python source -> the escaped form it takes inside a notebook string literal.

    Cell 5 writes `eng_runner.py` from a single-quoted literal, so patching it means matching the
    escaped text. Doing the escaping here beats hand-writing backslashes into the patterns.
    """
    return block.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")


def set_src(cell, text):
    cell["source"] = text.splitlines(keepends=True)


# ----------------------------------------------------------------------------- the mutator cell
MUT_CELL = MARK + '''
# M2/M3: scaffold harvest -> formula-space Delta-M mutation -> two-pass f.z screen, in its own
# CPU process (the S1 casmi snapshot that produced z). Any failure leaves MUT = {} and the
# notebook behaves exactly like the 0.399 fork.
MUT_ON, MUT_FRAC, MUT_K, MUT_N1 = True, 0.30, {K}, 2000{CFGX}
MUT_DIR = '/kaggle/working/mut'; os.makedirs(MUT_DIR, exist_ok=True)
MUT_FILES = {}
MUT_FILES['delta_m_mutator.py'] = {DM_SRC}
MUT_FILES['mut_runner.py'] = {MR_SRC}{FILESX}
for _n, _s in MUT_FILES.items():
    open(os.path.join(MUT_DIR, _n), 'w').write(_s)

MUT = {}
if MUT_ON and PC and os.path.exists('/kaggle/working/pc_z.npz'):
    try:
        _t0 = time.time()
        _env = dict(os.environ, MUT_FRAC=str(MUT_FRAC), MUT_K=str(MUT_K), MUT_N1=str(MUT_N1)){ENVX}
        _cmd = [sys.executable, os.path.join(MUT_DIR, 'mut_runner.py'), V3CODE, POOL_DIR,
                '/kaggle/working/pc_lists.json', '/kaggle/working/pc_z.npz',
                '/kaggle/working/mut_lists.json', '4']
        if not IS_RERUN:
            _cmd.append(str(MUT_MAXM))
        rr = subprocess.run(_cmd, capture_output=True, text=True, timeout=2 * 3600, env=_env)
        print(rr.stdout[-3000:]); print(rr.stderr[-2000:])
        MUT = json.load(open('/kaggle/working/mut_lists.json'))
    except Exception as e:
        print('MUTATOR FAILED -> 0.399 behaviour kept:', repr(e))
else:
    print('mutator skipped (MUT_ON=%s, PC=%d, pc_z=%s)'
          % (MUT_ON, len(PC), os.path.exists('/kaggle/working/pc_z.npz')))

# ---- sanity checks: shape, exact formulas, novelty, and no overlap with what we already have
_bad = []
for _m, _v in MUT.items():
    if not (len(_v['smiles']) == len(_v['keys']) == len(_v['formula'])):
        _bad.append(f'{_m}: ragged lists')
    if len(set(_v['keys'])) != len(_v['keys']):
        _bad.append(f'{_m}: duplicate keys')
    if len(_v['smiles']) > MUT_K:
        _bad.append(f'{_m}: {len(_v["smiles"])} > MUT_K')
    _pk = set(PC.get(_m, {}).get('pc_keys', []))
    if _pk & set(_v['keys']):
        _bad.append(f'{_m}: overlaps its own PubChem list'){SANX}
_n_mut = sum(len(v['smiles']) for v in MUT.values())
print(f'MUTATOR: {len(MUT)} molecules, {_n_mut} candidates, {_n_mut/max(1,len(MUT)):.1f} per molecule'
      f' | sanity {"OK" if not _bad else _bad[:5]}', flush=True)
if _bad:                      # never assert in a scoring run: a broken channel must cost 0, not the run
    print('MUTATOR SANITY FAILED -> channel disabled, 0.399 behaviour kept:', _bad[:10])
    MUT = {}{SRCX}
print(f'mutator {time.time()-T0:.0f}s')
'''

# --------------------------------------------------------------------- the --bio deltas
BIO_CFG = '''
MUT_DEADLINE = 45             # generation wall-clock per molecule: two sources are ~1.5x the pairs'''

BIO_ENV = '''
        _env['MUT_DEADLINE'] = str(MUT_DEADLINE)
        _engf = '/kaggle/working/eng_lists.json'
        if ENG and os.path.exists(_engf):   # BIO+AFIX scaffolds; absent -> PubChem only, as in v4n-mutator
            _env['MUT_ENG'] = _engf
        print('mutator scaffold sources: PubChem top-25' + (' + engine-A top-40' if 'MUT_ENG' in _env else ''))'''

BIO_SANITY = '''
    _ek = set((ENG.get(_m) or {}).get('keys', []))
    if _ek & set(_v['keys']):
        _bad.append(f'{_m}: overlaps its own engine list')
    if len(_v.get('src', _v['smiles'])) != len(_v['smiles']):
        _bad.append(f'{_m}: ragged src')'''

BIO_REPORT = '''
_n_eng = sum(sum(1 for x in _v.get('src', []) if x == 'eng') for _v in MUT.values())
_m_eng = sum(1 for _v in MUT.values() if 'eng' in _v.get('src', []))
print(f'MUT SOURCES (listed): {_n_eng} of {_n_mut} candidates grew from an engine-only scaffold,'
      f' across {_m_eng} molecules', flush=True)'''

# --------------------------------------------------------------------- the --c3 deltas
# The gate replaces MUT_FRAC entirely: it fires on a conjunction of similarities in [0,1] rather
# than on the weakest 30% by `best_pool_fz`, so firing on nothing is a legal outcome.  C3_MAX_STEPS
# is exposed here because the two bands worth probing differ only by it (measured on pseudo_c3,
# restricted to harvestable scaffolds: depth <= 1 fires on 2.5% with q3|fired 31.4%, depth <= 2 on
# 11.8% with 7.6% -- the same expected gain from four times the exposure).
C3_CFG = '''
MUT_DEADLINE = 45             # generation wall-clock per molecule, generation only
C3_MAX_STEPS = 1              # analog -> target delta depth; 1 is the measured band, 2 the probe'''

C3_ENV = '''
        _env['MUT_DEADLINE'] = str(MUT_DEADLINE)
        _env['C3_MAX_STEPS'] = str(C3_MAX_STEPS)
        _engf = '/kaggle/working/eng_lists.json'
        if ENG and os.path.exists(_engf):   # BIO+AFIX scaffolds; absent -> PubChem only
            _env['MUT_ENG'] = _engf
        _c3f = _engf + '.c3.json'           # engine Class-3 diagnostics; absent -> the old rank gate
        if os.path.exists(_c3f):
            _env['MUT_C3'] = _c3f
        print('mutator scaffold sources: PubChem top-25'
              + (' + engine-A top-40' if 'MUT_ENG' in _env else '')
              + (' + spectral analogs' if 'MUT_C3' in _env else '')
              + (' | Class-3 gate at depth <= %d' % C3_MAX_STEPS if 'MUT_C3' in _env
                 else ' | NO c3 diagnostics: falling back to the MUT_FRAC rank gate'))'''

C3_FILES = '''
MUT_FILES['c3_gate.py'] = {C3_SRC}'''

C3_REPORT = '''
_tiers = {}
for _v in MUT.values():
    _tiers[_v.get('tier') or 'none'] = _tiers.get(_v.get('tier') or 'none', 0) + 1
_n_ana = sum(sum(1 for x in _v.get('src', []) if x == 'ana') for _v in MUT.values())
print(f'MUT SOURCES (listed): tiers {_tiers} | {_n_ana} of {_n_mut} candidates grew from an'
      f' analog-only scaffold', flush=True)'''


def mut_cell(variant):
    """Fill the template. The pc-only variant substitutes empty strings, so its cell is unchanged.

    `variant` is "pc", "bio" or "c3".  c3 is a superset of bio -- it keeps the engine-A scaffold
    source and adds the Class-3 gate, the analog source and the per-candidate tier.
    """
    bio, c3 = variant in ("bio", "c3"), variant == "c3"
    return (MUT_CELL.replace('{K}', '40' if bio else '25')
            .replace('{CFGX}', (C3_CFG if c3 else BIO_CFG) if bio else '')
            .replace('{ENVX}', (C3_ENV if c3 else BIO_ENV) if bio else '')
            .replace('{SANX}', BIO_SANITY if bio else '')
            .replace('{SRCX}', (C3_REPORT if c3 else BIO_REPORT) if bio else '')
            .replace('{FILESX}', C3_FILES if c3 else ''))


def main():
    flags = {"--bio", "--c3"}
    argv = [a for a in sys.argv[1:] if a not in flags]
    c3 = "--c3" in sys.argv[1:]
    bio = c3 or "--bio" in sys.argv[1:]          # c3 keeps every bio delta and adds the gate
    variant = "c3" if c3 else ("bio" if bio else "pc")
    src = argv[0] if len(argv) > 0 else SRC
    dst = argv[1] if len(argv) > 1 else (DST_C3 if c3 else (DST_BIO if bio else DST))
    nb = json.load(open(src, encoding="utf-8"))
    cells = [c for c in nb["cells"] if MARK not in src_of(c) and "## Mutator channel" not in src_of(c)]
    if len(cells) != len(nb["cells"]):
        print(f"removed {len(nb['cells']) - len(cells)} previously injected cell(s)")
    nb["cells"] = cells

    def find(pred, what):
        for i, c in enumerate(nb["cells"]):
            if c["cell_type"] == "code" and pred(src_of(c)):
                return i
        raise SystemExit(f"PATCH FAILED: could not locate {what}")

    i_cfg = find(lambda s: "TOPN, ICE_LAM, ICE_BUDGET, ICE_PC" in s, "config cell")
    i_eng = find(lambda s: "eng_runner.py" in s, "engine cell")
    i_smoke = find(lambda s: "SMOKE_N = 12" in s, "commit-switch cell")
    i_pc = find(lambda s: "pc_runner.py" in s and "RUNNER" in s, "PubChem cell")
    i_ice = find(lambda s: "ICE_UNION" in s, "ICEBERG cell")
    i_fuse = find(lambda s: "def fuse2" in s, "fusion cell")

    # ---- 1. config: budgets up for the extra union candidates, plus MUT_MAXM default
    t = src_of(nb["cells"][i_cfg])
    t = must_replace(t, "TOPN, ICE_LAM, ICE_BUDGET, ICE_PC = 60, 1.0, 5400, True",
                     "TOPN, ICE_LAM, ICE_BUDGET, ICE_PC = 60, 1.0, 7200, True   # +1800s for the mutator union\n"
                     "MUT_MAXM, GL_BUDGET_SMOKE = 0, 200                        # the smoke cell overrides both",
                     "ICE_BUDGET")
    set_src(nb["cells"][i_cfg], t)

    # ---- 2. commit switch: cap the mutator and shorten GLACIER too (the fork only shortens ICE)
    t = src_of(nb["cells"][i_smoke])
    t = must_replace(t, "    ICE_BUDGET = 300\n",
                     "    ICE_BUDGET = 300\n    GL_BUDGET_SMOKE = 200\n    MUT_MAXM = 3\n",
                     "smoke budgets")
    set_src(nb["cells"][i_smoke], t)

    # ---- 2b. --c3 only: export the engine's Class-3 diagnostics alongside its lists.
    # Everything the gate needs is already in `recs`: `lib` is the per-candidate exact-mass library
    # similarity over the +-10 ppm window, and `ana` is the top-200 (pool index, similarity) over
    # +-200 Da -- AFIX mass-shifted analog propagation.  POOL carries the structure and the mass,
    # so the analog's scaffold and the exact mass shift are both recoverable.  This touches only
    # the notebook's own embedded runner, never casmi/engine.py or any MANIFEST-locked file.
    if c3:
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
                         esc(blk) + esc("    json.dump(res, open(out, 'w'))\n"),
                         "c3 diagnostics export")
        set_src(nb["cells"][i_eng], t)

    # ---- 3. M1: export the z logits pc_runner already has in RAM
    t = src_of(nb["cells"][i_pc])
    anchor = "    json.dump(res, open(OUT, \\'w\\'))\\n"
    add = ("    np.savez_compressed(os.path.join(os.path.dirname(OUT), \\'pc_z.npz\\'),\\n"
           "                        mids=np.array([t[0] for t in tasks], dtype=object),\\n"
           "                        z=np.stack([t[2] for t in tasks]).astype(np.float32))\\n")
    t = must_replace(t, anchor, add + anchor, "pc_z export")
    set_src(nb["cells"][i_pc], t)

    # ---- 4. M3: mutator candidates join the ICEBERG/GLACIER input (R2: every scorer sees them)
    t = src_of(nb["cells"][i_ice])
    anchor = "    if ICE_PC:"
    block = '''    if MUT:                              # ADDED (ICE_UNION_MUT): the mutator's candidates too.
        n_mut_add = 0                    # mandatory, not an optimisation -- generated structures have
        for m in mids:                   # no retrieval score, so ICE/GL are the only signal that can
            u = MUT.get(str(m))          # order them, and they are same-formula regioisomers.
            if not u or not u.get('smiles'):
                continue
            _known = set(BASE[m][1][:TOPN]) | set((ENG.get(str(m)) or {}).get('keys', []))
            _ms = [s for s, k in zip(u['smiles']{ICECAP}, u['keys']{ICECAP}) if k not in _known]
            _mf = [f for f, k in zip(u['formula']{ICECAP}, u['keys']{ICECAP}) if k not in _known]
            if not _ms:
                continue
            _before = len(cands.get(m, []))
            cands[m] = list(dict.fromkeys(list(cands.get(m, [])) + ice_fuse.ice_candidates(_ms, _mf, len(_ms))))
            n_mut_add += len(cands[m]) - _before
        print('ICE_UNION_MUT: mutator candidates added to ICE input:', n_mut_add, flush=True)
'''
    # MUT_K is 40 under --bio so engine-derived products add rather than displace, but only the
    # top 25 enter the forward simulators, so their load is identical to the PubChem-only notebook.
    block = block.replace("{ICECAP}", "[:25]" if bio else "")
    t = must_replace(t, anchor, block + anchor, "ICE_UNION_MUT")
    # a NameError on MUT would be raised inside the ICE try/except, which would silently disable
    # ICEBERG *and* leave `items` undefined so GLACIER dies too -- a >0.03 regression. Guard it.
    t = must_replace(t, "import gc\n", "import gc\nMUT = globals().get('MUT') or {}\n", "MUT guard")
    t = must_replace(t, "GL_LAM, GL_BUDGET = {}, dict(molecules=0, changed_vs_ice_top25=0, changed_vs_ice_top1=0), 1.0, 4000",
                      "GL_LAM, GL_BUDGET = {}, dict(molecules=0, changed_vs_ice_top25=0, changed_vs_ice_top1=0), 1.0, 5200",
                      "GL_BUDGET")
    t = must_replace(t, "GL_SCORES = gl_fuse.run_gl(GL_PKG, items, workdir='/kaggle/working/gl_work', device='cuda', budget_s=GL_BUDGET,",
                      "GL_SCORES = gl_fuse.run_gl(GL_PKG, items, workdir='/kaggle/working/gl_work', device='cuda',\n"
                      "                               budget_s=(GL_BUDGET_SMOKE if not IS_RERUN else GL_BUDGET),",
                      "GL smoke budget")
    set_src(nb["cells"][i_ice], t)

    # ---- 5. M4: third RRF term
    t = src_of(nb["cells"][i_fuse])
    t = must_replace(t, "ALPHA, KRR = 0.6, 3.0",
                     "ALPHA, KRR, ALPHA_MUT = 0.6, 3.0, 0.25   # mutator below engine A: least trustworthy source",
                     "ALPHA_MUT")
    t = must_replace(t, """def fuse2(v_smis, e_smis, e_keys, n=40):
    sc, smi_of = {}, {}
    for r, s in enumerate(v_smis, 1):
        k = chem.score_key(s) or s
        sc[k] = sc.get(k, 0.0) + 1.0 / (KRR + r); smi_of.setdefault(k, s)
    for r, (s, k) in enumerate(zip(e_smis, e_keys), 1):
        if not k:
            continue
        sc[k] = sc.get(k, 0.0) + ALPHA / (KRR + r); smi_of.setdefault(k, s)
    order = sorted(sc, key=lambda k: -sc[k])[:n]
    return [smi_of[k] for k in order], order, [sc[k] for k in order]""",
                     """def fuse2(v_smis, e_smis, e_keys, m_smis=(), m_keys=(), n=40):
    sc, smi_of = {}, {}
    for r, s in enumerate(v_smis, 1):
        k = chem.score_key(s) or s
        sc[k] = sc.get(k, 0.0) + 1.0 / (KRR + r); smi_of.setdefault(k, s)
    for r, (s, k) in enumerate(zip(e_smis, e_keys), 1):
        if not k:
            continue
        sc[k] = sc.get(k, 0.0) + ALPHA / (KRR + r); smi_of.setdefault(k, s)
    for r, (s, k) in enumerate(zip(m_smis, m_keys), 1):      # ADDED: mutator list, weight ALPHA_MUT
        if not k:
            continue
        sc[k] = sc.get(k, 0.0) + ALPHA_MUT / (KRR + r); smi_of.setdefault(k, s)
    order = sorted(sc, key=lambda k: -sc[k])[:n]
    return [smi_of[k] for k in order], order, [sc[k] for k in order]""", "fuse2 signature")
    guard = ("if ENG or MUT:\n    v4 = pd.read_csv('submission.csv')\n"
             "    changed = n_ice = 0; out = []\n    mut_keys_all = {m: set(v['keys']) for m, v in MUT.items()}\n"
             "    n_mut_in_top25 = n_mut_mols = 0")
    if c3:
        # ranks reserved for the mutator per tier.  Almost all displacement risk sits in slot 1: a
        # retrieved truth at rank 1 pushed to rank 4 loses 0.75 RR, every deeper shift an order of
        # magnitude less.  So `sure` takes slot 1 only once GC.2 has measured the pipeline's top-1
        # accuracy on the fired set; until then point both tiers at the hedged pattern.
        guard += ("\n    C3_SLOTS = {'sure': [1, 2, 3], 'likely': [2, 3, 4]}"
                  "\n    n_c3 = n_c3_ice = 0; n_c3_tier = {}")
    t = must_replace(t, "if ENG:\n    v4 = pd.read_csv('submission.csv')\n    changed = n_ice = 0; out = []",
                     guard, "fusion guard")
    if c3:
        # the mutator is deliberately absent from this fusion.  It emits same-formula regioisomers
        # of the target, so in the bio notebook it always landed in the correct answer's own
        # formula group and the z(RRF)+z(ICE)+z(GL) rerank could swap the two -- one such event
        # costs 0.95 RR, which is the whole of the observed 0.399 -> 0.397.  Promotion happens
        # below instead, by reserved rank, and only where the Class-3 gate was confident.
        call = """        e = ENG.get(str(mid)) or {'smiles': [], 'keys': []}
        u = MUT.get(str(mid)) or {'smiles': [], 'keys': []}
        _tier = (u.get('tier') or '') if u.get('smiles') else ''
        if e['smiles']:
            fsm, fk, fsc = fuse2(vs, e['smiles'], e['keys'])"""
    else:
        call = """        e = ENG.get(str(mid)) or {'smiles': [], 'keys': []}
        u = MUT.get(str(mid)) or {'smiles': [], 'keys': []}
        if e['smiles'] or u['smiles']:
            fsm, fk, fsc = fuse2(vs, e['smiles'], e['keys'], u['smiles'], u['keys'])"""
    t = must_replace(t, """        e = ENG.get(str(mid))
        if e and e['smiles']:
            fsm, fk, fsc = fuse2(vs, e['smiles'], e['keys'])""", call, "fuse2 call")
    if c3:
        after = """            changed += int(bool(vs) and fsm[:1] != vs[:1])
            vs = fsm[:25]
        if _tier:
            _ms, _mk = list(u['smiles']), list(u['keys'])
            _ice = ICE_SCORES.get(str(mid), {}) if isinstance(ICE_SCORES, dict) else {}
            _gl = gl_of(mid)
            if _ice:          # order the mutator's own candidates by forward-sim agreement. This is
                _sc = []      # legitimate where the fusion above was not: it ranks same-formula
                for _a, _b in zip(_ms, _mk):   # isomers against each other, never against a
                    _v = _ice.get(_b, _ice.get(_a, 0.0)) or 0.0      # retrieved candidate (R4).
                    if _gl:
                        _v += _gl.get(_b, _gl.get(_a, 0.0)) or 0.0
                    _sc.append(_v)
                if any(_sc):  # all-zero means neither key space matched, so keep the f.z order
                    _o = sorted(range(len(_ms)), key=lambda i: -_sc[i])
                    _ms = [_ms[i] for i in _o]; _mk = [_mk[i] for i in _o]
                    n_c3_ice += 1
            vs = merge(vs, [chem.score_key(x) or x for x in vs], _ms, _mk,
                       C3_SLOTS.get(_tier) or C3_SLOTS['likely'])
            n_c3 += 1; n_c3_tier[_tier] = n_c3_tier.get(_tier, 0) + 1
            _hit = sum(1 for x in vs[:25] if (chem.score_key(x) or x) in set(_mk))
            n_mut_in_top25 += _hit; n_mut_mols += int(_hit > 0)"""
    else:
        after = """            changed += int(bool(vs) and fsm[:1] != vs[:1])
            vs = fsm[:25]
            _mk = mut_keys_all.get(str(mid), set())      # how many mutator candidates actually landed
            if _mk:
                _hit = sum(1 for s in vs if (chem.score_key(s) or s) in _mk)
                n_mut_in_top25 += _hit; n_mut_mols += int(_hit > 0)"""
    t = must_replace(t, """            changed += int(bool(vs) and fsm[:1] != vs[:1])
            vs = fsm[:25]""", after, "mutator top-25 counter")
    rpt = ("    print('fused (+ICE) submission written; top-1 changed in', changed, '| ICE reordered', n_ice, f'{time.time()-T0:.0f}s')\n"
           "    print(f'MUTATOR IN FINAL TOP-25: {n_mut_in_top25} candidates across {n_mut_mols} molecules')")
    if c3:
        rpt += ("\n    print(f'C3 ROUTER: promoted {n_c3} molecules {n_c3_tier} | forward-sim "
                "reordered the mutator list on {n_c3_ice} | suppressed on {len(MUT) - n_c3}')")
    t = must_replace(t, "    print('fused (+ICE) submission written; top-1 changed in', changed, '| ICE reordered', n_ice, f'{time.time()-T0:.0f}s')",
                     rpt, "mutator report")
    set_src(nb["cells"][i_fuse], t)

    # ---- 6. insert the mutator cell right after the PubChem cell (PC and pc_z.npz exist there)
    dm = open(os.path.join(HERE, "delta_m_mutator.py"), encoding="utf-8").read()
    mr = open(os.path.join(HERE, "mut_runner.py"), encoding="utf-8").read()
    body = mut_cell(variant).replace("{DM_SRC}", repr(dm)).replace("{MR_SRC}", repr(mr))
    if c3:
        cg = open(os.path.join(HERE, "c3_gate.py"), encoding="utf-8").read()
        body = body.replace("{C3_SRC}", repr(cg))
    nb["cells"].insert(i_pc + 1, code(body))
    if c3:
        note = (
            "## Mutator channel \u2014 Class-3 routed (specs/track1_retrieval_fusion \u00a710)\n"
            "Bemis-Murcko scaffolds from each molecule's PubChem top-25, its engine-A top-40 "
            "(BIO = ChEBI + LIPID MAPS, AFIX analog propagation) **and its mass-shifted spectral "
            "analogs**, decorated to a target formula by the 20-transform biogenetic palette in "
            "formula space, screened by the same two-pass `f.z` the PubChem channel uses.\n\n"
            "Two things differ from `casmi26-v4n-mutator-bio-submit.ipynb` (LB 0.397):\n\n"
            "1. **It fires on a Class-3 judgement, not on a fixed 30% of the field.** `c3_gate` "
            "wants a strong mass-shifted analog, a weak exact-mass match, and a formula delta "
            "within `C3_MAX_STEPS` transforms of that analog. All three are similarities in [0,1] "
            "or counts, so none depends on the `f.z` scale \u2014 which matters because a commit "
            "run scores a slice of `train.parquet` and cannot calibrate an absolute `f.z` cut. "
            "Firing on zero molecules is a legal outcome.\n"
            "2. **Promotion replaces fusion.** Generated structures are same-formula regioisomers "
            "of the target, so in the bio notebook they always landed in the correct answer's own "
            "formula group, where `z(RRF)+z(ICE)+z(GL)` could swap the two \u2014 one such event "
            "costs 0.95 RR, the whole of 0.399 \u2192 0.397. Here the mutator is absent from the "
            "fusion and is spliced into ranks reserved by tier instead, and only where the gate "
            "fired. On a true Class-3 molecule the retrieval list holds no truth, so those ranks "
            "cost nothing.\n\n"
            "Measured on `p1_novel/pseudo_c3.parquet` (600 leak-free Class-3 natural products), "
            "restricted to scaffolds that are actually harvestable: depth \u2264 1 fires on 2.5% "
            "with a 80.0% chance the truth is generated, against 11.8% and 28.2% at depth "
            "\u2264 2 \u2014 the same expected gain from four times the exposure. Any failure "
            "leaves the 0.399 pipeline untouched.")
    else:
        note = (
            "## Mutator channel (specs_v4n_update)\n"
            "Bemis-Murcko scaffolds from "
            + ("each molecule's PubChem top-25 **and its engine-A top-40** (BIO = ChEBI + LIPID MAPS, "
               "AFIX analog propagation) \u2014 harvested separately and walked round-robin so neither "
               "source starves the other \u2014 decorated"
               if bio else "each molecule's PubChem top-25, decorated")
            + " to one of those hits' exact formulas by a 20-transform biogenetic palette in formula "
            "space (depth 6), screened by the same two-pass `f.z` the PubChem channel uses. Fires on "
            "the weakest 30% of molecules by `best_pool_fz`. "
            + ("Keeps 40 per molecule so engine-derived products add to the PubChem-derived ones rather "
               "than displacing them; only the top 25 enter the forward simulators. " if bio else "")
            + "Joins the ICEBERG/GLACIER union and the RRF at weight 0.25. "
            "Any failure leaves the 0.399 pipeline untouched.")
    nb["cells"].insert(i_pc + 1, md(note))

    json.dump(nb, open(dst, "w", encoding="utf-8"), indent=1)
    print(f"wrote {dst}  ({len(nb['cells'])} cells, "
          + ("PubChem + engine-A + analog scaffolds, Class-3 routed" if c3
             else ("PubChem + engine-A scaffolds" if bio else "PubChem scaffolds only")) + ")")
    return dst


if __name__ == "__main__":
    main()
