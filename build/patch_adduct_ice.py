"""Exploit probe generator: ICEBERG rerank for molecules GLACIER never sees (adduct-partitioned simulator).

Idea: research/exploit/20261010-adduct-iceberg-rerank (spec.md, verdict.md). Base: casmi-notebook-0-433.ipynb.
Never hand-edit the generated .ipynb -- change this script and regenerate. The source notebook is read, never
written. Outputs are cleared so the probe uploads cleanly.

GLACIER scores [M+H]+ spectra only, and base ICEBERG fusion is off (b421: ICE + GLACIER on the same molecule cost
0.015). Molecules with no [M+H]+ spectrum therefore get no simulator term in the BASE and fused reranks. This
probe switches ICEBERG back on for exactly those molecules:
  * eligible = no [M+H]+ spectrum AND >= 1 spectrum in BASE_ICE_ADDUCTS = {[M-H]-, [M+Na]+}   (str ids)
  * ICE cell:    base top-TOPN joins the ICE input (fallback when ENG lacks the molecule); BASE-list rerank by
                 z(score) + ICE_LAM z(ICE) for eligible ids with live ICE scores
  * fusion cell: the fused-list rerank gets the real ICE column instead of {} for the same ids
GLACIER emptiness is never the gate: if GLACIER fails, every [M+H]+ molecule has no GL scores, and gating on that
would rebuild the b420 configuration. BASE_ICE and CELL19_ICE stay False.

usage: python build/patch_adduct_ice.py [--src in.ipynb] [--out out.ipynb]
"""
from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "casmi-notebook-0-433.ipynb")
OUT = os.path.join(ROOT, "casmi26-adduct-ice.ipynb")


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


# ---------------------------------------------------------------- pure eligibility (also exec'd by the tests)
FUNCS = r'''
def adduct_ice_mids(te, adducts):
    """str molecule ids with no [M+H]+ spectrum and >= 1 spectrum whose adduct is in `adducts`."""
    want, out = set(adducts), set()
    for m, a in te.groupby('molecule_id', sort=False)['adduct']:
        s = set(a.astype(str))
        if '[M+H]+' not in s and s & want:
            out.add(str(m))
    return out


def adduct_ice_on(mid, ice, gl):
    """ICE rerank applies: eligible id, live ICE scores, and no GLACIER scores (eligible ids never have any)."""
    return (str(mid) in ADDUCT_ICE_MIDS and bool(ice) and any(v is not None for v in ice.values())
            and not gl)
'''

ELIG_BLOCK = r'''# === ADDUCT_ICE (patch_adduct_ice.py): ICEBERG rerank for molecules GLACIER never sees ===
BASE_ICE_ADDUCTS = {'[M-H]-', '[M+Na]+'}
ADDUCT_ICE_STATS = dict(eligible=0, cands_added=0, base_reranked=0, base_changed_top25=0,
                        fused_reranked=0, fused_changed_top25=0)
@FUNCS@
try:
    ADDUCT_ICE_MIDS = adduct_ice_mids(te, BASE_ICE_ADDUCTS)
except Exception as _e:
    ADDUCT_ICE_MIDS = set()
    print('[ADDUCT_ICE] eligibility failed -> off:', repr(_e), flush=True)
ADDUCT_ICE_STATS['eligible'] = len(ADDUCT_ICE_MIDS)
print('[ADDUCT_ICE] eligible molecules', len(ADDUCT_ICE_MIDS), flush=True)
'''

FUSE_GUARD = r'''# === ADDUCT_ICE (patch_adduct_ice.py): fused-list ICE for the same molecules; ICE cell failed early -> off ===
if 'adduct_ice_on' not in globals():
    ADDUCT_ICE_MIDS = set()
    ADDUCT_ICE_STATS = dict(eligible=0, cands_added=0, base_reranked=0, base_changed_top25=0,
                            fused_reranked=0, fused_changed_top25=0)
    def adduct_ice_on(mid, ice, gl):
        return False
'''


def patch_ice(t):
    t = must_replace(t, "\ntry:\n    ICE_PKG = ", "\n" + ELIG_BLOCK.replace("@FUNCS@", FUNCS) + "try:\n    ICE_PKG = ",
                     "eligibility block")
    t = must_replace(t,
                     "    cands = {m: (ice_fuse.ice_candidates(BASE[m][0], BASE[m][4], TOPN) if BASE_ICE else []) for m in mids}\n",
                     "    cands = {m: (ice_fuse.ice_candidates(BASE[m][0], BASE[m][4], TOPN)\n"
                     "                 if (BASE_ICE or str(m) in ADDUCT_ICE_MIDS) else []) for m in mids}\n"
                     "    ADDUCT_ICE_STATS['cands_added'] = 0 if BASE_ICE else sum(len(cands[m]) for m in mids if str(m) in ADDUCT_ICE_MIDS)\n",
                     "ICE input candidates")
    t = must_replace(t, "    if BASE_ICE and ice and any(v is not None for v in ice.values()):\n",
                     "    _aice = (not BASE_ICE) and adduct_ice_on(m, ice, gl)\n"
                     "    if (BASE_ICE or _aice) and ice and any(v is not None for v in ice.values()):\n",
                     "base-list rerank gate")
    t = must_replace(t, "            smis, keys = s2, k2\n",
                     "            if _aice:\n"
                     "                ADDUCT_ICE_STATS['base_reranked'] += 1\n"
                     "                ADDUCT_ICE_STATS['base_changed_top25'] += int(s2[:25] != smis[:25])\n"
                     "            smis, keys = s2, k2\n",
                     "base-list counters")
    return t


def patch_fuse(t):
    t = must_replace(t, "\nFUSION_OVERFLOW = {}\n", "\n" + FUSE_GUARD + "FUSION_OVERFLOW = {}\n", "fusion guard")
    t = must_replace(t, "            ice = ICE_SCORES.get(str(mid), {}) if isinstance(ICE_SCORES, dict) else {}\n",
                     "            ice = ICE_SCORES.get(str(mid), {}) if isinstance(ICE_SCORES, dict) else {}\n"
                     "            _aice = adduct_ice_on(mid, ice, gl_of(mid))\n",
                     "fused eligibility")
    t = must_replace(t, "[_form2(x) for x in fsm], (ice if CELL19_ICE else {}), lam=ICE_LAM, top_n=len(fsm))\n",
                     "[_form2(x) for x in fsm], (ice if (CELL19_ICE or _aice) else {}), lam=ICE_LAM, top_n=len(fsm))\n",
                     "fused ICE column")
    t = must_replace(t, "                    n_ice += int(o[:25] != list(range(min(25, len(o)))))\n",
                     "                    n_ice += int(o[:25] != list(range(min(25, len(o)))))\n"
                     "                    if _aice:\n"
                     "                        ADDUCT_ICE_STATS['fused_reranked'] += 1\n"
                     "                        ADDUCT_ICE_STATS['fused_changed_top25'] += int(o[:25] != list(range(min(25, len(o)))))\n",
                     "fused counters")
    t = must_replace(t, "\nprint('V1 inference complete; applying separately audited final pass next.')\n",
                     "\nprint('[ADDUCT_ICE] eligible', ADDUCT_ICE_STATS['eligible'], '| base_reranked', ADDUCT_ICE_STATS['base_reranked'],\n"
                     "      '| fused_reranked', ADDUCT_ICE_STATS['fused_reranked'], '| molecules reranked',\n"
                     "      ADDUCT_ICE_STATS['base_reranked'] + ADDUCT_ICE_STATS['fused_reranked'],\n"
                     "      '| changed_top25', ADDUCT_ICE_STATS['base_changed_top25'] + ADDUCT_ICE_STATS['fused_changed_top25'],\n"
                     "      '| candidates added', ADDUCT_ICE_STATS['cands_added'], flush=True)\n"
                     "print('V1 inference complete; applying separately audited final pass next.')\n",
                     "counter report")
    return t


def build(src=SRC):
    nb = json.load(open(src, encoding="utf-8"))
    i_ice = find(nb, lambda s: "BASE_ICE = False" in s and "ice_fuse.run_ice" in s, "ICE cell")
    i_fuse = find(nb, lambda s: "def fuse2" in s, "fusion cell")
    set_src(nb["cells"][i_ice], patch_ice(src_of(nb["cells"][i_ice])))
    set_src(nb["cells"][i_fuse], patch_fuse(src_of(nb["cells"][i_fuse])))
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
        c.get("metadata", {}).pop("papermill", None)
    return nb


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=SRC)
    p.add_argument("--out", default=OUT)
    a = p.parse_args(argv)
    nb = build(a.src)
    json.dump(nb, open(a.out, "w", encoding="utf-8"), indent=1)
    print("wrote", os.path.relpath(a.out, ROOT))
    return a.out


if __name__ == "__main__":
    main()
