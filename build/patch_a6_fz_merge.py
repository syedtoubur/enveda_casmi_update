"""A6 probe: f.z common-scale merge of PubChem and pool candidates in C9 (replaces slot merge + promotion).

Base: casmi26-e006-window-40-ice-fallback-true.ipynb (E006 v1, read only). Two edits:
  1. RUNNER patch (also used by patch_harness_dump.py --a6-runner): the embedded pc_runner.py string additionally
     records, per molecule, `pool_fz` = {pool key: raw f.z} for the top 300 pool rows of the +-10 ppm window
     (fallback 30 ppm), and `pc_rawfz` = RAW f.z of every PubChem-only candidate (probe_one's pc_fz is the z-scored
     score incl. popularity, NOT comparable to pool f.z). Right after `PC = json.load(...)` the notebook gets
     `p['_fzmap'] = {pc key: raw f.z}`, built before the ICE rerank reorders pc/pc_keys.
  2. C9 cell: knob FZ_MERGE (None = base; a float = margin in raw f.z units). When on and the row has pool_fz and
     pc_rawfz, fz_merge replaces merge/promote (the lib_max >= LIB_TAU lock stays). Any exception -> base result
     for that molecule. Prints `FZ_MERGE {"fz_merge_mols":..,"changed":..,"nofz":..,"margin":..}`.
Never hand-edit the generated .ipynb; change this script (or casmi_post.py) and regenerate.

usage: python build/patch_a6_fz_merge.py [--src in.ipynb] [--out out.ipynb] [--margin 50 | none]
"""
from __future__ import annotations

import argparse
import ast
import inspect
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
SRC = os.path.join(ROOT, "casmi26-e006-window-40-ice-fallback-true.ipynb")
OUT = os.path.join(ROOT, "casmi26-e006-a6-fz-merge.ipynb")

POOL_TOP = 300

# (old, new) edits on the decoded RUNNER source; each anchor must occur exactly once
RUNNER_EDITS = [
    ("    tasks, extra = [], {}\n",
     "    tasks, extra, zs = [], {}, {}   # A6: z kept for the raw f.z of the PubChem list\n"),
    ("    pmass = pd.read_parquet(os.path.join(POOL_DIR, 'pool_meta.parquet'), columns=['mass']).mass.values.astype(np.float64)\n",
     "    pmass = pd.read_parquet(os.path.join(POOL_DIR, 'pool_meta.parquet'), columns=['mass']).mass.values.astype(np.float64)\n"
     "    pkeys = pd.read_parquet(os.path.join(POOL_DIR, 'pool_meta.parquet'), columns=['key']).key.values   # A6\n"),
    ("        if b > a:\n            F = np.unpackbits(",
     "        pool_fz = {}   # A6\n        if b > a:\n            F = np.unpackbits("),
    ("            best = float((F @ z.astype(np.float64)).max())\n",
     "            fzv = F @ z.astype(np.float64)\n"
     "            best = float(fzv.max())\n"
     "            for _r in np.argsort(-fzv, kind='stable')[:%d]:   # A6: top rows by f.z, best per key\n"
     "                _k = pkeys[a + int(_r)]\n"
     "                if _k not in pool_fz and np.isfinite(fzv[_r]):\n"
     "                    pool_fz[_k] = float(fzv[_r])\n"
     "                    if len(pool_fz) >= %d:\n"
     "                        break\n" % (POOL_TOP * 3, POOL_TOP)),
    ("        extra[mid] = dict(best_pool_fz=best, target=target)\n        tasks.append((mid, target, z))\n",
     "        extra[mid] = dict(best_pool_fz=best, target=target, pool_fz=pool_fz)\n        tasks.append((mid, target, z)); zs[mid] = z\n"),
    ("top_pop=d.get('top_pop', 0.0), **extra[mid])\n",
     "top_pop=d.get('top_pop', 0.0), **extra[mid])\n"
     "            try:   # A6: RAW f.z of each PubChem-only candidate (pc_fz is the z-scored score), same z as the pool rows\n"
     "                res[mid]['pc_rawfz'] = [float(chem.raw_fingerprint(_s)[bits].astype(np.float32) @ zs[mid]) for _s in smis]\n"
     "            except Exception as _e:\n"
     "                print('ERROR pc_rawfz', mid, repr(_e), flush=True); res[mid]['pc_rawfz'] = None\n"),
]

# notebook side, right after the PC load
LOAD_ANCHOR = "    PC = json.load(open('/kaggle/working/pc_lists.json'))\n"
LOAD_NEW = (LOAD_ANCHOR +
            "    for _p in PC.values():   # A6: raw f.z per PubChem key, before the ICE rerank reorders pc/pc_keys\n"
            "        if isinstance(_p, dict) and _p.get('pc_rawfz') is not None and len(_p['pc_rawfz']) == len(_p.get('pc_keys', [])):\n"
            "            _p['_fzmap'] = dict(zip(_p['pc_keys'], _p['pc_rawfz']))\n")


def src_of(cell):
    return "".join(cell["source"])


def must_replace(text, old, new, what):
    if text.count(old) != 1:
        raise SystemExit(f"PATCH FAILED: anchor for {what} occurs {text.count(old)} times:\n  {old[:120]}")
    return text.replace(old, new)


def patch_runner_text(runner: str) -> str:
    for old, new in RUNNER_EDITS:
        runner = must_replace(runner, old, new, "RUNNER " + old.strip()[:40])
    compile(runner, "pc_runner.py", "exec")
    return runner


def patch_runner_cell(text: str, with_load_hook: bool = True) -> str:
    """The cell holding `RUNNER = '<literal>'` (one line) and the PC load. Patches the decoded string, re-emits repr."""
    lines = text.split("\n")
    idx = [i for i, ln in enumerate(lines) if ln.startswith("RUNNER = ")]
    if len(idx) != 1:
        raise SystemExit(f"PATCH FAILED: RUNNER literal lines: {len(idx)}")
    runner = ast.literal_eval(lines[idx[0]][len("RUNNER = "):])
    lines[idx[0]] = "RUNNER = " + repr(patch_runner_text(runner))
    out = "\n".join(lines)
    if with_load_hook:
        out = must_replace(out, LOAD_ANCHOR, LOAD_NEW, "PC load")
    return out


# ---- C9 cell
A_KNOB = ("S_TAU, POP_TAU = 6.0, 5.0\n",
          "S_TAU, POP_TAU = 6.0, 5.0\nFZ_MERGE = @M@   # A6 margin (raw f.z units); None = base merge/promote\n")
A_STATS = ("dict(untouched=0, gentle=0, aggressive=0, no_pc=0, promoted=0)",
           "dict(untouched=0, gentle=0, aggressive=0, no_pc=0, promoted=0, fz_merge_mols=0, fz_merge_changed_vs_base=0, "
           "fz_merge_nofz=0, fz_merge_err=0)")
A_TAIL = ("            final = merge(smis, keys, p['pc'], p['pc_keys'], slots)\n    if not final: final = ['CCO']\n",
          "            final = merge(smis, keys, p['pc'], p['pc_keys'], slots)\n"
          "        if FZ_MERGE is not None:\n"
          "            try:\n"
          "                _fm = fz_merge_row(p, smis, keys, float(FZ_MERGE))\n"
          "            except Exception:\n"
          "                _fm = None; stats['fz_merge_err'] += 1\n"
          "            if _fm is None:\n"
          "                stats['fz_merge_nofz'] += 1\n"
          "            else:\n"
          "                stats['fz_merge_mols'] += 1; stats['fz_merge_changed_vs_base'] += int(_fm[:25] != final[:25])\n"
          "                _sl = next((q for q, s in enumerate(_fm, 1) if s == p['pc'][0]), 0)\n"
          "                stats['fz_merge_pc1_slot_%d' % _sl] = stats.get('fz_merge_pc1_slot_%d' % _sl, 0) + 1\n"
          "                final = _fm\n"
          "    if not final: final = ['CCO']\n")
A_PRINT = ("print('merge stats', stats)\n",
           "print('merge stats', stats)\n"
           "print('FZ_MERGE', json.dumps(dict(fz_merge_mols=stats['fz_merge_mols'], changed=stats['fz_merge_changed_vs_base'], "
           "nofz=stats['fz_merge_nofz'], err=stats['fz_merge_err'], margin=FZ_MERGE)), flush=True)\n")
A_DEF = ("rows, stats = [], dict(", "@FUNCS@\nrows, stats = [], dict(")


def funcs_src() -> str:
    import casmi_post
    return inspect.getsource(casmi_post.fz_merge) + "\n\n" + inspect.getsource(casmi_post.fz_merge_row)


def patch_c9_text(text: str, margin) -> str:
    text = must_replace(text, A_KNOB[0], A_KNOB[1].replace("@M@", repr(None if margin is None else float(margin))), "FZ_MERGE knob")
    text = must_replace(text, *A_STATS, "stats dict")
    text = must_replace(text, *A_TAIL, "C9 tail")
    text = must_replace(text, *A_PRINT, "merge stats print")
    text = must_replace(text, A_DEF[0], A_DEF[1].replace("@FUNCS@", funcs_src()), "function defs")
    compile(text, "c9", "exec")
    return text


def build(src: str = SRC, margin=50.0) -> dict:
    nb = json.load(open(src, encoding="utf-8"))
    cells = nb["cells"]
    hit = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and "RUNNER = '" in src_of(c) and "best = float((F @ z" in src_of(c)]
    if len(hit) != 1:
        raise SystemExit(f"PATCH FAILED: runner cell matched {len(hit)} cells")
    cells[hit[0]]["source"] = patch_runner_cell(src_of(cells[hit[0]])).splitlines(keepends=True)
    c9 = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and "(rel > 250.0)" in src_of(c)]
    if len(c9) != 1:
        raise SystemExit(f"PATCH FAILED: C9 cell matched {len(c9)} cells")
    cells[c9[0]]["source"] = patch_c9_text(src_of(cells[c9[0]]), margin).splitlines(keepends=True)
    return nb


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--margin", default="50", help="raw f.z margin, or 'none' for knob off")
    a = ap.parse_args(argv)
    nb = build(a.src, None if a.margin.lower() == "none" else float(a.margin))
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
