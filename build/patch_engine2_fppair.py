"""engine2-fppair probe: E006 v1 + Engine 2 "ours" ranker on sim_rank_rows_fppair.npz with leak-free FPNets, w_pv 0.50.

Base: casmi26-e006-window-40-ice-fallback-true.ipynb (E006 v1, read only). Only the Engine 2 cell is rewritten: the
embedded casmi_engine.py / eng_runner.py sources (ENG_FILES literal and the runner write) get the edits below, and the
cell passes the two knobs (ENGINE2_ROWS, ENGINE2_W_PV, defaults in build/casmi_post.py CFG) to the runner by env.
Every edit is gated: with ENGINE2_ROWS == 'nofp' the engine path is the base path (nofp rows, use_fp=False, w_pv 0.88).
Fallback (files missing, FPNet width != pool fp width, unknown block, width guard): prints `E2_ROWS FALLBACK <reason>`
and runs the base path at w_pv 0.88, so the nofp ranker can never ship at 0.50 weight.
Never hand-edit the generated .ipynb; change this script and regenerate.

usage: python build/patch_engine2_fppair.py [--src in.ipynb] [--out out.ipynb] [--rows fppair|nofp] [--w-pv 0.50]
"""
from __future__ import annotations

import argparse
import ast
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "casmi26-e006-window-40-ice-fallback-true.ipynb")
OUT = os.path.join(ROOT, "casmi26-e006-engine2-fppair.ipynb")
ENG_LINE = "ENG_FILES = "
RUNNER_LINE = "open(os.path.join(ENG_DIR, 'eng_runner.py'), 'w').write("


def must_replace(text, old, new, what):
    if text.count(old) != 1:
        raise SystemExit(f"PATCH FAILED: anchor for {what} occurs {text.count(old)} times:\n  {old[:120]}")
    return text.replace(old, new)


# ---------------------------------------------------------------------------------------------- casmi_engine.py
E2_HELPERS = '''_E2T = {"fp": 0.0}


def _e2p(msg):
    """Engine 2 counter: stdout, and a file the notebook cell re-prints (the cell only shows the log tail)."""
    print(msg, flush=True)
    try:
        with open("/kaggle/working/e2_counters.txt", "a") as f:
            f.write(msg + "\\n")
    except OSError:
        pass


def _e2_load(our_fp_paths, dev):
    """Leak-free FPNets for the "ours" ranker -> (models, None) or (None, fallback reason)."""
    import pv_fp
    try:
        s2, m2, nb2 = pv_fp.load_fp_models(our_fp_paths, dev)
    except Exception as ex:
        return None, f"load {ex!r}"[:200]
    wp = int(pool_fps(np.zeros(1, np.int64)).shape[1])
    if not (s2 or m2): return None, "no FPNet loaded"
    if nb2 != wp: return None, f"FPNet nbits {nb2} != candidate fp width {wp}"
    return (s2, m2, dev), None


def _e2_check(recs, fppair_path):
    """Schema guard (L25) on the fppair rows: blocks known, hstack width == X.shape[1], leak-free logits non-zero.
    -> (ok, reason). Never truncates the "ours" matrix."""
    z = np.load(fppair_path, allow_pickle=True)
    blocks = [str(b) for b in z["blocks"]]
    width = int(z["X"].shape[1])
    unknown = [b for b in blocks if b not in BLOCKS]
    if unknown: return False, f"unknown blocks {unknown}"
    cands = [r for r in recs if len(r["cand"]) and r.get("zlog_ours") is not None]
    nz = sum(1 for r in cands if np.any(r["zlog_ours"] != 0))
    n_act = sum(1 for r in recs if len(r["cand"]))
    frac = nz / max(1, n_act)
    _e2p(f"E2_FP ours models=2 nonzero_frac={frac:.3f} molecules={n_act}")
    if not cands or nz == 0: return False, "zlog_ours missing or all zero"
    r = cands[0]
    rr = dict(r, zlog=r["zlog_ours"])
    w = int(np.hstack([BLOCKS[b](rr, pool_fps(r["cand"])) for b in blocks]).shape[1])
    if w != width: return False, f"hstack width {w} != X.shape[1] {width}"
    _e2p(f"E2_ROWS fppair rows={int(z['X'].shape[0])} feats={width} blocks={blocks}")
    return True, None


def _e2_plan(recs, models2, fppair_path, our_rank_path, w_pv, w_pv_fp):
    """-> (rank_path, use_fp, zkey, w_pv). Any failed check = the base call args (nofp rows, use_fp=False, w_pv as given)."""
    if models2 is not None:
        ok, why = _e2_check(recs, fppair_path)
        if ok: return fppair_path, True, "zlog_ours", w_pv_fp
        _e2p(f"E2_ROWS FALLBACK {why}")
    return our_rank_path, False, "zlog", w_pv


'''

ENGINE_EDITS = [
    ("def compute_channels(L, I, models, te):",
     E2_HELPERS + "def compute_channels(L, I, models, te, models2=None):", "compute_channels signature"),
    ("                                         ces, modes)\n            u, lmax, lmean, lsh, lcnt = lib",
     "                                         ces, modes)\n"
     "            zlog2 = None\n"
     "            if models2 is not None:                  # engine2-fppair: leak-free FPNets, own field (pv never sees it)\n"
     "                _t2 = time.time()\n"
     "                zlog2 = pv_fp.molecule_logits(models2, specs, sub.precursor_mz.values.astype(np.float32),\n"
     "                                              list(sub.adduct.astype(str).values),\n"
     "                                              list(sub.instrument_type.astype(str).values), ces, modes)\n"
     "                _E2T[\"fp\"] += time.time() - _t2\n"
     "            u, lmax, lmean, lsh, lcnt = lib", "zlog_ours compute"),
    ("zlog=None if zlog is None else zlog.astype(np.float32), specs=specs,",
     "zlog=None if zlog is None else zlog.astype(np.float32),\n"
     "                       zlog_ours=None if zlog2 is None else zlog2.astype(np.float32), specs=specs,", "rec zlog_ours"),
    ("def score_molecules(recs, rankers, blocks, use_fp):",
     "def score_molecules(recs, rankers, blocks, use_fp, zkey=\"zlog\"):", "score_molecules signature"),
    ("        rr = r if use_fp else dict(r, zlog=None)",
     "        rr = (r if zkey == \"zlog\" else dict(r, zlog=r[zkey])) if use_fp else dict(r, zlog=None)", "score_molecules zkey"),
    ("workers=4, limit=0, w_pv=0.88):",
     "workers=4, limit=0, w_pv=0.88,\n         our_fp_paths=None, fppair_path=None, w_pv_fp=0.50):", "main signature"),
    ("    mols, recs = compute_channels(L, I, (single, merged, dev), te)\n",
     "    models2 = None\n"
     "    if our_fp_paths and fppair_path:\n"
     "        models2, why = _e2_load(our_fp_paths, dev)\n"
     "        if why: _e2p(f\"E2_ROWS FALLBACK {why}\")\n"
     "    mols, recs = compute_channels(L, I, (single, merged, dev), te, models2=models2)\n", "main compute_channels call"),
    ("    ours, blocks = fit_rankers(our_rank_path)\n"
     "    s_ours = score_molecules(recs, ours, blocks, use_fp=False)\n",
     "    rank_path, use_fp, zkey, w_pv = _e2_plan(recs, models2, fppair_path, our_rank_path, w_pv, w_pv_fp)\n"
     "    ours, blocks = fit_rankers(rank_path)\n"
     "    s_ours = score_molecules(recs, ours, blocks, use_fp=use_fp, zkey=zkey)\n", "main ours scoring"),
    ("    return subs, recs, {\"pv\": s_pv, \"ours\": s_ours}",
     "    return subs, recs, {\"pv\": s_pv, \"ours\": s_ours, \"w\": w_pv}", "main return"),
]

RUNNER_EDITS = [
    ("    T0 = time.time()\n    subs, recs, S = E.main(test, train, sample, E.find('sim_rank_rows_nofp.npz'), E.find('rank_train.npz'), fp_models,\n"
     "                           workers=os.cpu_count(), w_pv=0.88)\n"
     "    bl = E.blend_scores(S['pv'], S['ours'], 0.88)\n",
     "    T0 = time.time()\n"
     "    try: os.remove('/kaggle/working/e2_counters.txt')\n"
     "    except OSError: pass\n"
     "    e2_rows, e2_w = os.environ.get('E2_ROWS', 'nofp'), float(os.environ.get('E2_W_PV', '0.88'))\n"
     "    kw = {}\n"
     "    if e2_rows == 'fppair':\n"
     "        try:\n"
     "            kw = dict(fppair_path=E.find('sim_rank_rows_fppair.npz'), w_pv_fp=e2_w,\n"
     "                      our_fp_paths=[E.find('ours_fpnet_single_16k.pt'), E.find('ours_fpnet_merged_26k.pt')])\n"
     "        except FileNotFoundError as ex:\n"
     "            E._e2p(f'E2_ROWS FALLBACK missing {ex}')\n"
     "    subs, recs, S = E.main(test, train, sample, E.find('sim_rank_rows_nofp.npz'), E.find('rank_train.npz'), fp_models,\n"
     "                           workers=os.cpu_count(), w_pv=0.88, **kw)\n"
     "    bl = E.blend_scores(S['pv'], S['ours'], S['w'])\n"
     "    E._e2p(f\"E2_BLEND w_pv={S['w']:.2f}\")\n", "runner main call"),
    ("    print('engine lists', len(res), f'{time.time()-T0:.0f}s', flush=True)\n",
     "    print('engine lists', len(res), f'{time.time()-T0:.0f}s', flush=True)\n"
     "    E._e2p(f\"E2_TIME fp_ours={E._E2T['fp']:.0f}s total={time.time()-T0:.0f}s\")\n", "runner counters"),
]

CELL_EDITS = [
    ("ENG = {}\ntry:\n",
     "ENGINE2_ROWS, ENGINE2_W_PV = @ROWS@, @WPV@   # engine2-fppair knobs (build/casmi_post.py CFG defaults: 'nofp', 0.88)\n"
     "ENG = {}\ntry:\n", "knob constants"),
    ("timeout=4 * 3600, cwd=ENG_DIR)",
     "timeout=4 * 3600, cwd=ENG_DIR,\n"
     "                        env=dict(os.environ, E2_ROWS=ENGINE2_ROWS, E2_W_PV=str(ENGINE2_W_PV)))", "runner env"),
    ("    print('engine lists', len(ENG), f'{time.time()-_t0:.0f}s')\n",
     "    print('engine lists', len(ENG), f'{time.time()-_t0:.0f}s')\n"
     "    try: print(open('/kaggle/working/e2_counters.txt').read())\n"
     "    except OSError: print('E2 counters file missing')\n", "counter re-print"),
]


def patch_source(text, rows="fppair", w_pv=0.50):
    lines = text.split("\n")
    ie = [i for i, l in enumerate(lines) if l.startswith(ENG_LINE)]
    ir = [i for i, l in enumerate(lines) if l.startswith(RUNNER_LINE)]
    if len(ie) != 1 or len(ir) != 1:
        raise SystemExit(f"PATCH FAILED: ENG_FILES lines {len(ie)}, runner lines {len(ir)}")
    files = ast.literal_eval(ast.parse(lines[ie[0]]).body[0].value)
    runner = ast.literal_eval(ast.parse(lines[ir[0]]).body[0].value.args[0])
    eng = files["casmi_engine.py"]
    for old, new, what in ENGINE_EDITS:
        eng = must_replace(eng, old, new, what)
    for old, new, what in RUNNER_EDITS:
        runner = must_replace(runner, old, new, what)
    files["casmi_engine.py"] = eng
    lines[ie[0]] = ENG_LINE + repr(files)
    lines[ir[0]] = RUNNER_LINE + repr(runner) + ")"
    text = "\n".join(lines)
    for old, new, what in CELL_EDITS:
        text = must_replace(text, old, new.replace("@ROWS@", repr(rows)).replace("@WPV@", repr(w_pv)), what)
    return text


def build(src=SRC, rows="fppair", w_pv=0.50):
    nb = json.load(open(src, encoding="utf-8"))
    hits = [c for c in nb["cells"] if c["cell_type"] == "code" and "".join(c["source"]).startswith("'''Our engine (two-ranker")]
    if len(hits) != 1:
        raise SystemExit(f"PATCH FAILED: Engine 2 cell matched {len(hits)} cells")
    cell = hits[0]
    cell["source"] = patch_source("".join(cell["source"]), rows, w_pv).splitlines(keepends=True)
    cell["outputs"], cell["execution_count"] = [], None
    return nb


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--rows", default="fppair", choices=("fppair", "nofp"))
    ap.add_argument("--w-pv", type=float, default=0.50)
    a = ap.parse_args(argv)
    nb = build(a.src, a.rows, a.w_pv)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
