"""B2 neutral-loss simulator similarity: E006 v1 + a third GLACIER-driven z-term (gl_nl_mean, weight NL_LAM = 0.5).

Base: casmi26-e006-window-40-ice-fallback-true.ipynb (E006 v1, read only). What changes (all default-safe):
  * cell 7 (ICEBERG + GLACIER): the GLACIER package is copied to /kaggle/working/gl_pkg_nl and its gl_runner.py is
    text-patched (exact anchors, each exactly once; compile-checked) so score_molecule ALSO computes the entropy
    similarity in neutral-loss space (NL = prec - mz, NL > 0.5 Da, ascending) from the prediction already in memory
    (no second forward pass) and writes {mid: {smiles: gl_nl_mean}} to <out>.nl.json. A failed patch prints
    B2_NL_PATCH_FAILED and GLACIER runs unpatched (NL_SCORES stays empty -> behaviour identical to E006 v1).
  * the same z-fusion that GLACIER drives gets nl as a THIRD dict: [ice, gl, nl] / [ICE_LAM, GL_LAM, NL_LAM] at the five
    notebook call sites (cell 7 gl_rerank x2 callers, cell 9 fused list, E004 overlay loop, E006 rerank_row). The
    E004-control reconstruction assert in the E006 overlay needs the SAME fusion upstream and in rerank_row, so every
    site that uses GL uses NL.
  * prints  B2_NL {"nl_scored": n, "nl_rerank_mols": n, "nl_lam": 0.5}.
Never hand-edit the generated .ipynb; change this script and regenerate.

usage: python build/patch_b2_nl_sim.py [--src in.ipynb] [--out out.ipynb] [--nl-lam 0.5]
"""
from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "casmi26-e006-window-40-ice-fallback-true.ipynb")
OUT = os.path.join(ROOT, "casmi26-e006-b2-nl-sim.ipynb")

# ------------------------------------------------------------------------------------------------ runner patch -----
# Source embedded verbatim in the notebook (and exec'd by the tests). Anchors are from the casmi26-glacier gl_runner.py.
NB_RUNNER_PATCH = r"""
B2_RUNNER_NL_HELPERS = '''
# ---- B2 neutral-loss similarity (patched in) ------------------------------------------------------------------
NL_MIN = 0.5
NL_RESULT = {}
_NL_LAST = {}


def nl_prep(mz, p, prec):
    # fragment spectrum (mz, weights) -> neutral-loss spectrum: NL = prec - mz, keep NL > NL_MIN Da, ascending,
    # weights renormalised to sum 1 (the precursor-adjacent peaks drop out).
    mz = np.asarray(mz, np.float64)
    p = np.asarray(p, np.float64)
    nl = float(prec) - mz
    k = nl > NL_MIN
    if not k.any():
        return np.empty(0, np.float64), np.empty(0, np.float64)
    nl, w = nl[k], p[k]
    o = np.argsort(nl, kind='stable')
    nl, w = nl[o], w[o]
    s = w.sum()
    if not s > 0:
        return np.empty(0, np.float64), np.empty(0, np.float64)
    return nl, w / s


'''
B2_RUNNER_ANCHORS = [
    ("        out.append((qm, qp, sp['adduct'], instr_token(sp), ce_bucket(spectrum_ce(sp))))\n",
     "        out.append((qm, qp, sp['adduct'], instr_token(sp), ce_bucket(spectrum_ce(sp)), float(sp['prec'])))\n"),
    ("    for _, _, ad, ins, ceb in plan:\n",
     "    for _, _, ad, ins, ceb, _prec in plan:\n"),
    ("def score_molecule(plan, cands, cache):\n",
     B2_RUNNER_NL_HELPERS + "def score_molecule(plan, cands, cache):\n    _NL_LAST.clear()\n"),
    ("    for c in cands:\n        sims = []\n        for qm, qp, ad, ins, ceb in plan:\n",
     "    for c in cands:\n        sims = []\n        nlsims = []\n        for qm, qp, ad, ins, ceb, prec in plan:\n"),
    ("            s = entropy_sim(qm, qp, pm, pp, TOL, PPM)\n",
     "            s = entropy_sim(qm, qp, pm, pp, TOL, PPM)\n"
     "            try:\n"
     "                qn, qw = nl_prep(qm, qp, prec)\n"
     "                pn, pw = nl_prep(pm, pp, prec)\n"
     "                sn = entropy_sim(qn, qw, pn, pw, TOL, PPM)\n"
     "                if np.isfinite(sn):\n"
     "                    nlsims.append(float(sn))\n"
     "            except Exception:\n"
     "                pass\n"),
    ("        res[c] = float(np.mean(sims)) if sims else None\n",
     "        res[c] = float(np.mean(sims)) if sims else None\n"
     "        _NL_LAST[c] = float(np.mean(nlsims)) if nlsims else None\n"),
    ("                        result[mid].update(sc)\n",
     "                        result[mid].update(sc)\n"
     "                        NL_RESULT.setdefault(mid, {}).update(_NL_LAST)\n"),
    ("                write_json(a.out, result)\n                log(f'chunk:",
     "                write_json(a.out, result)\n"
     "                write_json(a.out + '.nl.json', NL_RESULT)\n"
     "                log(f'chunk:"),
]


def b2_patch_runner_text(text):
    # exact-string patch of gl_runner.py text; every anchor must occur exactly once, else raise.
    text = text.replace('\r\n', '\n')
    for old, new in B2_RUNNER_ANCHORS:
        n = text.count(old)
        if n != 1:
            raise RuntimeError('B2 runner anchor occurs %d times: %r' % (n, old[:80]))
        text = text.replace(old, new)
    compile(text, 'gl_runner_b2', 'exec')
    return text


def b2_make_nl_pkg(pkg, dst='/kaggle/working/gl_pkg_nl'):
    import shutil
    if os.path.exists(dst):
        shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(pkg, dst, ignore=shutil.ignore_patterns('ckpt'))
    try:
        os.symlink(os.path.join(pkg, 'ckpt'), os.path.join(dst, 'ckpt'))
    except Exception:
        shutil.copytree(os.path.join(pkg, 'ckpt'), os.path.join(dst, 'ckpt'))
    f = os.path.join(dst, 'gl_runner.py')
    with open(f, newline='') as fh:
        t = fh.read()
    with open(f, 'w') as fh:
        fh.write(b2_patch_runner_text(t))
    return dst
"""


def _runner_ns():
    ns = {"os": os}
    exec(NB_RUNNER_PATCH, ns)
    return ns


def runner_patch_fn():
    return _runner_ns()["b2_patch_runner_text"]


def NB_RUNNER_ANCHORS_FOR_TEST():
    return _runner_ns()["B2_RUNNER_ANCHORS"]


# --------------------------------------------------------------------------------------------- notebook patches ---
def src_of(cell):
    return "".join(cell["source"])


def must_replace(text, old, new, what):
    if text.count(old) != 1:
        raise SystemExit(f"PATCH FAILED: anchor for {what} occurs {text.count(old)} times:\n  {old[:120]}")
    return text.replace(old, new)


A_CELL7_KEY = "gl_fuse.run_gl(GL_PKG, gl_items"
A_CELL9_KEY = "gl_fuse.rerank_multi(fsm, fk, fsc"
A_E004_KEY = "[{}, _gl], [ICE_LAM, GL_LAM]"
A_E006_KEY = 'scope["gl_fuse"].rerank_multi('

# cell 7
C7_STATE = ("GL_SCORES, gl_stats, GL_LAM, GL_BUDGET = {}, dict(molecules=0, changed_vs_ice_top25=0, changed_vs_ice_top1=0), 1.0, 4000\n",
            "GL_SCORES, gl_stats, GL_LAM, GL_BUDGET = {}, dict(molecules=0, changed_vs_ice_top25=0, changed_vs_ice_top1=0), 1.0, 4000\n"
            "'''B2: neutral-loss entropy similarity vs GLACIER predictions = third z-term (weight NL_LAM; None = off)'''\n"
            "NL_LAM, NL_SCORES, B2_MOLS = @NL_LAM@, {}, set()\n"
            "@RUNNER_PATCH@\n"
            "def nl_of(m):\n"
            "    g = NL_SCORES.get(str(m), {}) if (NL_LAM is not None and isinstance(NL_SCORES, dict)) else {}\n"
            "    if isinstance(g, dict) and any(v is not None for v in g.values()):\n"
            "        B2_MOLS.add(str(m))\n"
            "        return g\n"
            "    return {}\n"
            "def b2_report(tag=''):\n"
            "    import json as _j\n"
            "    n = sum(v is not None for d in NL_SCORES.values() if isinstance(d, dict) for v in d.values()) if isinstance(NL_SCORES, dict) else 0\n"
            "    print('B2_NL', _j.dumps(dict(nl_scored=n, nl_rerank_mols=len(B2_MOLS), nl_lam=NL_LAM)), tag, flush=True)\n")
C7_RUN = ("    GL_SCORES = gl_fuse.run_gl(GL_PKG, gl_items, workdir='/kaggle/working/gl_work', device='cuda', budget_s=GL_BUDGET,\n"
          "                               site='/kaggle/working/ice_site', wheels=os.path.join(ICE_PKG, 'wheels'), extra_args=['--any-rdkit'])\n",
          "    _GL_RUN_PKG = GL_PKG\n"
          "    if NL_LAM is not None:\n"
          "        try:\n"
          "            _GL_RUN_PKG = b2_make_nl_pkg(GL_PKG)\n"
          "            print('B2 runner patched ->', _GL_RUN_PKG, flush=True)\n"
          "        except Exception as _e:\n"
          "            _GL_RUN_PKG = GL_PKG\n"
          "            print('B2_NL_PATCH_FAILED', repr(_e), flush=True)\n"
          "    _nlf = '/kaggle/working/gl_work/gl_out.json.nl.json'\n"
          "    if os.path.exists(_nlf):\n"
          "        os.remove(_nlf)\n"
          "    GL_SCORES = gl_fuse.run_gl(_GL_RUN_PKG, gl_items, workdir='/kaggle/working/gl_work', device='cuda', budget_s=GL_BUDGET,\n"
          "                               site='/kaggle/working/ice_site', wheels=os.path.join(ICE_PKG, 'wheels'), extra_args=['--any-rdkit'])\n"
          "    if _GL_RUN_PKG != GL_PKG and os.path.exists(_nlf):\n"
          "        try:\n"
          "            NL_SCORES = json.load(open(_nlf))\n"
          "        except Exception as _e:\n"
          "            NL_SCORES = {}\n"
          "            print('B2_NL_PARSE_FAILED', repr(_e), flush=True)\n")
C7_SIG = ("def gl_rerank(order0, smis, keys, scs, forms, ice, gl, top_n, tag=''):\n",
          "def gl_rerank(order0, smis, keys, scs, forms, ice, gl, top_n, tag='', nlm=None):\n")
C7_CALL = ("        o = gl_fuse.rerank_multi(smis, keys, scs, forms, [ice, gl], [ICE_LAM, GL_LAM], top_n=top_n)\n",
           "        _nlg = nl_of(nlm) if nlm is not None else {}\n"
           "        o = gl_fuse.rerank_multi(smis, keys, scs, forms, [ice, gl] + ([_nlg] if _nlg else []),\n"
           "                                 [ICE_LAM, GL_LAM] + ([NL_LAM] if _nlg else []), top_n=top_n)\n")
C7_CALLER1 = ("order = gl_rerank(order, smis, keys, scs, forms, ice, gl, TOPN)",
              "order = gl_rerank(order, smis, keys, scs, forms, ice, gl, TOPN, nlm=m)")
C7_CALLER2 = ("o = gl_rerank(o, p['pc'], p['pc_keys'], p['pc_fz'], p['pc_form'], ice, gl, 25, 'pc_')",
              "o = gl_rerank(o, p['pc'], p['pc_keys'], p['pc_fz'], p['pc_form'], ice, gl, 25, 'pc_', nlm=m)")
C7_END = ("print('GL rerank stats', gl_stats, f'{time.time()-T0:.0f}s')",
          "print('GL rerank stats', gl_stats, f'{time.time()-T0:.0f}s')\nb2_report('after_cell7')")
# cell 9
C9_CALL = ("                        o = gl_fuse.rerank_multi(fsm, fk, fsc, [_form2(x) for x in fsm], [(ice if CELL19_ICE else {}), _g], [ICE_LAM, GL_LAM], top_n=len(fsm))\n",
           "                        _nlg = nl_of(mid)\n"
           "                        o = gl_fuse.rerank_multi(fsm, fk, fsc, [_form2(x) for x in fsm], [(ice if CELL19_ICE else {}), _g] + ([_nlg] if _nlg else []), [ICE_LAM, GL_LAM] + ([NL_LAM] if _nlg else []), top_n=len(fsm))\n")
C9_END = ("print('V1 inference complete; applying separately audited final pass next.')",
          "print('V1 inference complete; applying separately audited final pass next.')\nb2_report('after_cell9')")
# E004 overlay loop
E4_CALL = ("            _order = gl_fuse.rerank_multi(_candidates, _keys, _scores,\n"
           "                [_form2(s) for s in _candidates], [{}, _gl], [ICE_LAM, GL_LAM], top_n=len(_candidates))\n",
           "            _nlg = nl_of(_mid)\n"
           "            _order = gl_fuse.rerank_multi(_candidates, _keys, _scores,\n"
           "                [_form2(s) for s in _candidates], [{}, _gl] + ([_nlg] if _nlg else []), [ICE_LAM, GL_LAM] + ([NL_LAM] if _nlg else []), top_n=len(_candidates))\n")
# E006 rerank_row
E6_CALL = ('            order = scope["gl_fuse"].rerank_multi(\n'
           '                candidates,\n'
           '                keys,\n'
           '                scores,\n'
           '                [scope["_form2"](s) for s in candidates],\n'
           '                [{}, gl],\n'
           '                [scope["ICE_LAM"], scope["GL_LAM"]],\n'
           '                top_n=len(candidates),\n'
           '            )\n',
           '            _nlg = scope["nl_of"](mid)\n'
           '            order = scope["gl_fuse"].rerank_multi(\n'
           '                candidates,\n'
           '                keys,\n'
           '                scores,\n'
           '                [scope["_form2"](s) for s in candidates],\n'
           '                [{}, gl] + ([_nlg] if _nlg else []),\n'
           '                [scope["ICE_LAM"], scope["GL_LAM"]] + ([scope["NL_LAM"]] if _nlg else []),\n'
           '                top_n=len(candidates),\n'
           '            )\n')
E6_END = ("FACTORIAL_REPORT = _factorial_entry(globals())\ndel _factorial_entry",
          "FACTORIAL_REPORT = _factorial_entry(globals())\ndel _factorial_entry\nb2_report('final')")

CHANGED_KEYS = (A_CELL7_KEY, A_CELL9_KEY, A_E004_KEY, A_E006_KEY)


def _cell(nb, key):
    hits = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code" and key in src_of(c)]
    if len(hits) != 1:
        raise SystemExit(f"PATCH FAILED: cell for {key!r} matched {len(hits)} cells")
    return hits[0]


def build(src: str = SRC, nl_lam: float = 0.5) -> dict:
    nb = json.load(open(src, encoding="utf-8"))
    edits = {
        A_CELL7_KEY: [C7_STATE, C7_RUN, C7_SIG, C7_CALL, C7_CALLER1, C7_CALLER2, C7_END],
        A_CELL9_KEY: [C9_CALL, C9_END],
        A_E004_KEY: [E4_CALL],
        A_E006_KEY: [E6_CALL, E6_END],
    }
    for key, pairs in edits.items():
        i = _cell(nb, key)
        text = src_of(nb["cells"][i])
        for old, new in pairs:
            new = new.replace("@NL_LAM@", repr(float(nl_lam))).replace("@RUNNER_PATCH@", NB_RUNNER_PATCH.strip("\n"))
            text = must_replace(text, old, new, old[:50].strip())
        nb["cells"][i]["source"] = text.splitlines(keepends=True)
    return nb


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--nl-lam", type=float, default=0.5)
    a = ap.parse_args(argv)
    nb = build(a.src, a.nl_lam)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
