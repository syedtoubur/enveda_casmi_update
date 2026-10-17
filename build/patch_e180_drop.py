"""e180-drop-dreamsnp-pclite probe generator: patch V22 baseline notebook.

Base: casmi26-v22-rel150-e2r2-clean.ipynb (V22 clean, read only).
Patches Cell 34 (_factorial_entry) and Cell 36 (_e2r2_apply) so that:
  - E180_DROP filters e180-only synthetic decoys after simulator/FRAG_GAP reranking
    and before the 25-slot cut, backfilling to 25 from ranked overflow and mass-matched
    DreamsNP natural products.
  - E2R2 decoy guard prevents e180_only decoys from being inserted at Rank 2.
  - Cell 36 prints E180_B8 counters for value-checked verification.

Never hand-edit the generated .ipynb; change this script and regenerate.
Outputs are cleared so the generated notebook commits cleanly.

usage: python build/patch_e180_drop.py [--src in.ipynb] [--out out.ipynb] [--e180-drop [BOOL]] [--dreams-fill [BOOL]]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "scratch", "baseline", "casmi26-v22-rel150-e2r2-clean.ipynb")
OUT = os.path.join(ROOT, "casmi26-e180-drop-dreamsnp-pclite.ipynb")


def src_of(cell):
    return "".join(cell["source"])


def must_replace(text, old, new, what):
    if text.count(old) != 1:
        raise SystemExit(f"PATCH FAILED: anchor for {what} occurs {text.count(old)} times:\n  {old[:120]}")
    return text.replace(old, new)


# ---------------------------------------------------------------- Cell 34: _factorial_entry edits
C34_HEAD_OLD = '''    import hashlib
    import json
    import math
    from pathlib import Path'''

C34_HEAD_NEW = '''    import hashlib
    import json
    import math
    from pathlib import Path

    E180_DROP = @E180_DROP@
    DREAMS_FILL = @DREAMS_FILL@

    def _read_e180_keys(path):
        with open(path, encoding='utf-8') as f:
            return frozenset(line.strip() for line in f if line.strip())

    E180_KEY_PATHS = (
        '/kaggle/input/casmi26-e180-keys/e180_only_ik14.txt',
        'E:/casmi/data/e180/e180_only_ik14.txt',
    )

    def _get_e180_keys():
        if not E180_DROP:
            return None
        st_keys = scope.get('e180_keys')
        if st_keys is not None:
            return frozenset(st_keys)
        path = next((p for p in E180_KEY_PATHS if Path(p).is_file()), None)
        if path is None:
            raise FileNotFoundError(f'E180_DROP=True but no key file found (tried {E180_KEY_PATHS})')
        k = _read_e180_keys(path)
        if not k:
            raise ValueError('E180_DROP=True but key set is empty')
        return k

    _e180_keys = _get_e180_keys()

    def _is_e180(smi):
        if _e180_keys is None:
            return False
        ident = scope["metric_identity"](str(smi).strip())
        return ident is not None and ident[0] in _e180_keys

    _dreams_cache = None
    def _get_dreams():
        nonlocal _dreams_cache
        if not DREAMS_FILL:
            return None
        if _dreams_cache is not None:
            return _dreams_cache
        if 'dreams' in scope and scope['dreams'] is not None:
            _dreams_cache = scope['dreams']
            return _dreams_cache
        candidates = [
            Path('E:/casmi/data/dreams'),
            Path('/kaggle/input/casmi26-dreams-np'),
        ]
        for p in candidates:
            mp, sp = p / 'dreams_np_mass.npy', p / 'dreams_np_smiles.npy'
            if mp.is_file() and sp.is_file():
                import numpy as np
                _dreams_cache = (np.load(mp), np.load(sp))
                return _dreams_cache
        try:
            import glob
            matches = glob.glob('/kaggle/input/**/dreams_np_mass.npy', recursive=True)
            if matches:
                p = Path(matches[0]).parent
                mp, sp = p / 'dreams_np_mass.npy', p / 'dreams_np_smiles.npy'
                if mp.is_file() and sp.is_file():
                    import numpy as np
                    _dreams_cache = (np.load(mp), np.load(sp))
                    return _dreams_cache
        except Exception:
            pass
        return None'''

C34_RERANK_SIG_OLD = '    def rerank_row(scope, mid, original, *, window, ice_fallback, processed):'
C34_RERANK_SIG_NEW = '    def rerank_row(scope, mid, original, *, window, ice_fallback, processed, e180_drop=False):'

C34_RESULT_OLD = '''        result = (
            scope["safe_extend"](candidates[:25], candidates[25:] + original)
            if candidates
            else original
        )'''

C34_RESULT_NEW = '''        result = (
            scope["safe_extend"](candidates[:25], candidates[25:] + original)
            if candidates
            else original
        )
        if e180_drop and _e180_keys is not None and candidates:
            n = sum(_is_e180(s) for s in result[:25])
            if n:
                kept = [s for s in candidates if not _is_e180(s)]
                orig = [s for s in original if not _is_e180(s)]
                result = scope["safe_extend"](kept[:25], kept[25:] + orig) or result
                audit["e180_dropped"] = n'''

C34_LOOP_OLD = '''        output, audits = [], []
        for row in reference.itertuples(index=False):
            mid, original = str(row.molecule_id), str(row.smiles).split(";")
            active = mid in processed
            # The exact original pre-overlay tail is essential for paired equivalence.
            rebuilt, _ = rerank_row(
                scope, mid, original, window=40, ice_fallback=False, processed=active
            )
            assert ";".join(rebuilt) == controls[mid], f"E004 control reconstruction mismatch: {mid}"
            result, audit = rerank_row(
                scope, mid, original, window=window, ice_fallback=ice_fallback, processed=active
            )'''

C34_LOOP_NEW = '''        output, audits = [], []
        dropped_mol = {}
        dreams_added_total = 0
        _targets = scope['te'].groupby('molecule_id').nm.median().to_dict() if 'te' in scope else {}
        for row in reference.itertuples(index=False):
            mid, original = str(row.molecule_id), str(row.smiles).split(";")
            active = mid in processed
            # The exact original pre-overlay tail is essential for paired equivalence.
            rebuilt, _ = rerank_row(
                scope, mid, original, window=40, ice_fallback=False, processed=active, e180_drop=False
            )
            assert ";".join(rebuilt) == controls[mid], f"E004 control reconstruction mismatch: {mid}"
            result, audit = rerank_row(
                scope, mid, original, window=window, ice_fallback=ice_fallback, processed=active, e180_drop=E180_DROP
            )
            if E180_DROP and _e180_keys is not None and active:
                # rerank_row has already pruned its own top 25; count those drops too.
                n_pre = audit.get("e180_dropped", 0) if isinstance(audit, dict) else 0
                n_k = sum(_is_e180(s) for s in result[:25])
                if n_k or n_pre:
                    dropped_mol[mid] = dropped_mol.get(mid, 0) + n_k + n_pre
                short = len(result) < 25 and (mid in dropped_mol or n_k > 0 or (DREAMS_FILL and _get_dreams() is not None))
                if n_k or short:
                    kept = [s for s in result if not _is_e180(s)]
                    _over = scope.get("FUSION_OVERFLOW", {}).get(mid, [])
                    _tail = scope.get("BASE_TAIL_CANDIDATES", {}).get(mid, [])
                    _eng = (scope.get("ENG", {}).get(mid) or {}).get("smiles", [])
                    _pc_entry = scope.get("PC", {}).get(mid, {})
                    _pc = _pc_entry.get("pc", []) if isinstance(_pc_entry, dict) else []
                    extras = list(_over) + list(_tail) + list(_eng) + list(_pc)
                    non_keyed_extras = [s for s in extras if not _is_e180(s)]
                    revised = scope["safe_extend"](kept, non_keyed_extras, limit=25)
                    dm = _get_dreams()
                    if DREAMS_FILL and dm is not None and len(revised) < 25:
                        t = float(_targets.get(mid, float('nan')))
                        if math.isfinite(t) and t > 0:
                            import numpy as np
                            dm_mass, dm_smiles = dm
                            order_ = np.argsort(dm_mass, kind='stable'); srt = dm_mass[order_]
                            lo, hi = np.searchsorted(srt, [t * (1 - 10e-6), t * (1 + 10e-6)])
                            ix = order_[lo:hi]
                            ix = ix[np.argsort(np.abs(dm_mass[ix] - t), kind='stable')]
                            dm_cands = [x.decode() if isinstance(x, bytes) else str(x) for x in dm_smiles[ix]]
                            dm_non_keyed = [s for s in dm_cands if not _is_e180(s)]
                            before_len = len(revised)
                            revised = scope["safe_extend"](revised, dm_non_keyed, limit=25)
                            dreams_added_total += len(revised) - before_len
                    result = revised or ['CCO']'''

C34_REPORT_OLD = '''            "same_run_control_reconstructed": True,
            "provenance": provenance,'''

C34_REPORT_NEW = '''            "same_run_control_reconstructed": True,
            "provenance": provenance,
            "e180_dropped": sum(dropped_mol.values()),
            "e180_mols": len(dropped_mol),
            "dreams_added": dreams_added_total,'''


# ---------------------------------------------------------------- Cell 36: _e2r2_apply edits
C36_E2R2_ROW_OLD = '''def e2r2_row(toks, eng_list, key, limit=25, placeholder='CCO'):
    """Put engine 2's top candidate x2 at rank 2 of one final list (best first). Pure function.

    x2 is the first entry of `eng_list` that is a non-blank string other than the placeholder. It is used only when
    key(x2) is not None; if that entry is unusable the row is left alone (later entries are not tried).
      empty row, or rank 1 is the placeholder  -> unchanged, 'skipped'
      no engine list                           -> unchanged, 'no_eng'
      x2 missing, unparseable or holds ';'     -> unchanged, 'invalid_x2'
      key(x2) == key(rank 1)                   -> unchanged, 'same_as_rank1'
      x2's identity already at rank 2          -> that token stays, deeper twins of it are dropped, 'already_rank2'
      otherwise                                -> every token with key(x2) is dropped, x2 goes to rank 2, the other
                                                  tokens keep their order, the list is cut to `limit`, 'inserted'
    Returns (new_toks, tag, moved_from). moved_from is 'absent' or the old 1-based rank (str) of x2's identity when
    x2 ends at rank 2, else None.
    """
    toks = list(toks)
    if not toks or not isinstance(toks[0], str) or toks[0].strip() in ('', placeholder):
        return toks, 'skipped', None
    if isinstance(eng_list, str):
        eng_list = [eng_list]
    if not eng_list:
        return toks, 'no_eng', None
    x2 = next((s.strip() for s in eng_list if isinstance(s, str) and s.strip() not in ('', placeholder)), None)'''

C36_E2R2_ROW_NEW = '''def e2r2_row(toks, eng_list, key, limit=25, placeholder='CCO', e180_keys=None):
    """Put engine 2's top candidate x2 at rank 2 of one final list (best first). Pure function.

    x2 is the first entry of `eng_list` that is a non-blank string other than the placeholder
    (and not in e180_keys when provided). It is used only when key(x2) is not None; if that entry
    is unusable the row is left alone (later entries are not tried).
      empty row, or rank 1 is the placeholder  -> unchanged, 'skipped'
      no engine list                           -> unchanged, 'no_eng'
      x2 missing, unparseable or holds ';'     -> unchanged, 'invalid_x2'
      key(x2) == key(rank 1)                   -> unchanged, 'same_as_rank1'
      x2's identity already at rank 2          -> that token stays, deeper twins of it are dropped, 'already_rank2'
      otherwise                                -> every token with key(x2) is dropped, x2 goes to rank 2, the other
                                                  tokens keep their order, the list is cut to `limit`, 'inserted'
    Returns (new_toks, tag, moved_from). moved_from is 'absent' or the old 1-based rank (str) of x2's identity when
    x2 ends at rank 2, else None.
    """
    toks = list(toks)
    if not toks or not isinstance(toks[0], str) or toks[0].strip() in ('', placeholder):
        return toks, 'skipped', None
    if isinstance(eng_list, str):
        eng_list = [eng_list]
    if not eng_list:
        return toks, 'no_eng', None
    if e180_keys is not None:
        def _keyed(s):
            k = key(s)
            return k is not None and k[0] in e180_keys
        eng_list = [s for s in eng_list if not _keyed(s)]
        if not eng_list:
            return toks, 'no_eng', None
    x2 = next((s.strip() for s in eng_list if isinstance(s, str) and s.strip() not in ('', placeholder)), None)'''

C36_APPLY_LOOP_OLD = '''    check(base)
    tags, moved, rows, examples = Counter(), Counter(), [], []
    eligible_active = tail_dropped = twins_dropped = 0
    for mid, s in zip(base.molecule_id, base.smiles):
        old = s.split(';')
        new, tag, moved_from = e2r2_row(old, engine_list(mid), key)'''

C36_APPLY_LOOP_NEW = '''    check(base)
    _e180_keys = None
    if @E180_DROP@:
        _e180_paths = (
            '/kaggle/input/casmi26-e180-keys/e180_only_ik14.txt',
            'E:/casmi/data/e180/e180_only_ik14.txt',
        )
        _ep = next((p for p in _e180_paths if os.path.exists(p)), None)
        if _ep is not None:
            with open(_ep, encoding='utf-8') as _f:
                _e180_keys = frozenset(l.strip() for l in _f if l.strip())

    tags, moved, rows, examples = Counter(), Counter(), [], []
    eligible_active = tail_dropped = twins_dropped = e180_blocked = 0
    for mid, s in zip(base.molecule_id, base.smiles):
        old = s.split(';')
        elist = engine_list(mid)
        if _e180_keys is not None and elist:
            raw_x2 = next((s.strip() for s in elist if isinstance(s, str) and s.strip() not in ('', 'CCO')), None)
            if raw_x2 is not None:
                kx2 = key(raw_x2)
                if kx2 is not None and kx2[0] in _e180_keys:
                    e180_blocked += 1
        new, tag, moved_from = e2r2_row(old, elist, key, e180_keys=_e180_keys)'''

C36_PRINT_OLD = '''    print('E2R2_APPLIED', json.dumps(stats), flush=True)
    for ex in examples:
        print('E2R2_EXAMPLE', json.dumps(ex), flush=True)
    return stats'''

C36_PRINT_NEW = '''    print('E2R2_APPLIED', json.dumps(stats), flush=True)
    for ex in examples:
        print('E2R2_EXAMPLE', json.dumps(ex), flush=True)
    _fac = scope.get('FACTORIAL_REPORT', {})
    print('E180_B8', json.dumps({
        'e180_dropped': _fac.get('e180_dropped', 0),
        'e180_mols': _fac.get('e180_mols', 0),
        'e2r2_e180_blocked': e180_blocked,
        'dreams_added': _fac.get('dreams_added', 0)
    }), flush=True)
    return stats'''


def patch_cell34(text: str, e180_drop: bool = True, dreams_fill: bool = True) -> str:
    text = must_replace(text, C34_HEAD_OLD,
                        C34_HEAD_NEW.replace("@E180_DROP@", repr(bool(e180_drop))).replace("@DREAMS_FILL@", repr(bool(dreams_fill))),
                        "Cell 34 head knobs")
    text = must_replace(text, C34_RERANK_SIG_OLD, C34_RERANK_SIG_NEW, "Cell 34 rerank_row signature")
    text = must_replace(text, C34_RESULT_OLD, C34_RESULT_NEW, "Cell 34 result pruning")
    text = must_replace(text, C34_LOOP_OLD, C34_LOOP_NEW, "Cell 34 loop and DreamsNP backfill")
    text = must_replace(text, C34_REPORT_OLD, C34_REPORT_NEW, "Cell 34 report counters")
    return text


def patch_cell36(text: str, e180_drop: bool = True) -> str:
    text = must_replace(text, C36_E2R2_ROW_OLD, C36_E2R2_ROW_NEW, "Cell 36 e2r2_row decoy guard")
    text = must_replace(text, C36_APPLY_LOOP_OLD,
                        C36_APPLY_LOOP_NEW.replace("@E180_DROP@", repr(bool(e180_drop))),
                        "Cell 36 apply loop")
    text = must_replace(text, C36_PRINT_OLD, C36_PRINT_NEW, "Cell 36 print E180_B8 counters")
    return text


def build(src: str = SRC, e180_drop: bool = True, dreams_fill: bool = True) -> dict:
    nb = json.load(open(src, encoding="utf-8"))
    c34_hits = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code" and "def _factorial_entry(scope):" in src_of(c)]
    if len(c34_hits) != 1:
        raise SystemExit(f"PATCH FAILED: Cell 34 matched {len(c34_hits)} cells")
    c36_hits = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code" and "def _e2r2_apply(scope," in src_of(c)]
    if len(c36_hits) != 1:
        raise SystemExit(f"PATCH FAILED: Cell 36 matched {len(c36_hits)} cells")

    i34, i36 = c34_hits[0], c36_hits[0]
    nb["cells"][i34]["source"] = patch_cell34(src_of(nb["cells"][i34]), e180_drop, dreams_fill).splitlines(keepends=True)
    nb["cells"][i36]["source"] = patch_cell36(src_of(nb["cells"][i36]), e180_drop).splitlines(keepends=True)

    # Clear outputs so the generated notebook commits cleanly
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"] = []
            c["execution_count"] = None

    return nb


def parse_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Patch V22 baseline notebook for e180-drop-dreamsnp-pclite")
    ap.add_argument("--src", default=SRC, help="Source baseline notebook (.ipynb)")
    ap.add_argument("--out", default=OUT, help="Target patched notebook (.ipynb)")
    ap.add_argument("--e180-drop", type=parse_bool, nargs="?", const=True, default=True, help="Enable E180_DROP filter")
    ap.add_argument("--dreams-fill", type=parse_bool, nargs="?", const=True, default=True, help="Enable DreamsNP backfill")
    a = ap.parse_args(argv)

    nb = build(a.src, a.e180_drop, a.dreams_fill)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
