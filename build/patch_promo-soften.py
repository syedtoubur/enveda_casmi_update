"""promo-soften probe: E006 v1 with the C9 `rel` promotion bar raised 250 -> 500 (knob REL_PROMOTE).

Base: casmi26-e006-window-40-ice-fallback-true.ipynb (E006 v1, read only). Only the C9 merge/promotion cell changes:
  * `REL_PROMOTE = 500.0` next to `S_TAU, POP_TAU`,
  * the literal `(rel > 250.0)` becomes `(rel > REL_PROMOTE)`,
  * stats gets `promote_rel_skipped` (the base would have promoted: S/pop arm false, 250 < rel <= REL_PROMOTE),
    printed by the existing `print('merge stats', stats)`.
With --rel-promote 250 the cell decides exactly like the base (the counter stays 0).
Never hand-edit the generated .ipynb; change this script and regenerate.

usage: python build/patch_promo-soften.py [--src in.ipynb] [--out out.ipynb] [--rel-promote 500]
"""
from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "casmi26-e006-window-40-ice-fallback-true.ipynb")
OUT = os.path.join(ROOT, "casmi26-e006-promo-soften.ipynb")

A_KNOB = ("S_TAU, POP_TAU = 6.0, 5.0\n",
          "S_TAU, POP_TAU = 6.0, 5.0\nREL_PROMOTE = @REL@   # promo-soften: base literal was 250.0\n")
A_STATS = ("dict(untouched=0, gentle=0, aggressive=0, no_pc=0, promoted=0)",
           "dict(untouched=0, gentle=0, aggressive=0, no_pc=0, promoted=0, promote_rel_skipped=0)")
A_TEST = ("        if (S is not None and float(S) > S_TAU and pop_top >= POP_TAU) or (rel > 250.0):\n"
          "            stats['promoted'] += 1\n"
          "            final = promote(smis, keys, p['pc'], p['pc_keys'], slots)\n"
          "        else:\n"
          "            final = merge(smis, keys, p['pc'], p['pc_keys'], slots)\n",
          "        if (S is not None and float(S) > S_TAU and pop_top >= POP_TAU) or (rel > REL_PROMOTE):\n"
          "            stats['promoted'] += 1\n"
          "            final = promote(smis, keys, p['pc'], p['pc_keys'], slots)\n"
          "        else:\n"
          "            if 250.0 < rel <= REL_PROMOTE:\n"
          "                stats['promote_rel_skipped'] += 1\n"
          "            final = merge(smis, keys, p['pc'], p['pc_keys'], slots)\n")


def src_of(cell):
    return "".join(cell["source"])


def must_replace(text, old, new, what):
    if text.count(old) != 1:
        raise SystemExit(f"PATCH FAILED: anchor for {what} occurs {text.count(old)} times:\n  {old[:120]}")
    return text.replace(old, new)


def build(src: str = SRC, rel_promote: float = 500.0) -> dict:
    nb = json.load(open(src, encoding="utf-8"))
    hits = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code" and "(rel > 250.0)" in src_of(c)]
    if len(hits) != 1:
        raise SystemExit(f"PATCH FAILED: C9 cell matched {len(hits)} cells")
    text = src_of(nb["cells"][hits[0]])
    text = must_replace(text, A_KNOB[0], A_KNOB[1].replace("@REL@", repr(float(rel_promote))), "REL_PROMOTE knob")
    text = must_replace(text, *A_STATS, "stats dict")
    text = must_replace(text, *A_TEST, "promotion test")
    nb["cells"][hits[0]]["source"] = text.splitlines(keepends=True)
    return nb


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--rel-promote", type=float, default=500.0)
    a = ap.parse_args(argv)
    nb = build(a.src, a.rel_promote)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
