"""CPU tests for harness.py on a synthetic dump (fake fuse / gl_fuse / casmi.chem).  python research/harness/test_harness.py"""
from __future__ import annotations

import pickle
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import harness  # noqa: E402

SM = ['CCCCCCCO', 'CCCO', 'CCCCO', 'CCCCCO', 'CCCCCCO', 'CC(C)O', 'CC(C)CO', 'OCCO']


def make_dump(root: Path):
    src = root / "dumps" / "T" / "post_dump" / "src"
    (src / "casmi").mkdir(parents=True)
    (src / "fuse.py").write_text("def rerank(smis, keys, scs, forms, ice, lam=1.0, top_n=25):\n    return list(range(len(smis)))[:top_n]\n")
    (src / "gl_fuse.py").write_text("def rerank_multi(smis, keys, scs, forms, cols, lams, top_n=25):\n    return list(range(len(smis)))[::-1][:top_n]\n")
    (src / "casmi" / "__init__.py").write_text("")
    (src / "casmi" / "chem.py").write_text("def score_key(s):\n    return s\n")
    st = dict(base={'v1': (SM[:3], SM[:3], 0.95, [3.0, 2.0, 1.0], ['F'] * 3),
                    'v2': (SM[3:5], SM[3:5], 0.95, [2.0, 1.0], ['F'] * 2)},
              pc={}, eng={'v2': dict(smiles=[SM[5], SM[4]], keys=[SM[5], SM[4]])},
              ice={'v2': {'x': 1.0}}, gl={'v2': {'x': 1.0}}, te_mids=['v1', 'v2'], sample_ids=['v1', 'v2'],
              targets={}, consts={})
    pickle.dump(st, open(src.parent / "state.pkl", "wb"))
    return {'v1': dict(molecule_id='v1', smiles=SM[1], cls='1'), 'v2': dict(molecule_id='v2', smiles=SM[4], cls='2')}


def main() -> int:
    td = Path(tempfile.mkdtemp())
    try:
        tr = make_dump(td)
        harness.DUMPS = td / "dumps"
        harness.truths = lambda: tr
        r = harness.compare("T", {}, {"USE_ENG": False})
        # base: v1 truth at rank 2 (0.5); v2: fused list then GL reversal -> compute both and check consistency
        assert r["n"] == 2 and abs(r["mrr_a"] - r["mrr_b"] - (-r["delta"])) < 1e-9
        assert r["ci90"][0] <= r["delta"] <= r["ci90"][1]
        same = harness.compare("T", {}, {})
        assert same["delta"] == 0 and not same["lb_slot"]
        assert harness.parse_set("CELL19_ICE=True, ALPHA=0.5") == {"CELL19_ICE": True, "ALPHA": 0.5}
        # identity cache: persisted per dump, same result, and a reload never calls the dump's score_key again
        harness.save_caches()
        assert (td / "dumps" / "T" / "ident_cache.pkl").exists()
        (td / "dumps" / "T" / "post_dump" / "src" / "casmi" / "chem.py").write_text(
            "def score_key(s):\n    raise RuntimeError('cache miss')\n")
        sys.modules.pop("casmi.chem", None); sys.modules.pop("casmi", None)
        harness._CACHES.clear()
        again = harness.compare("T", {}, {"USE_ENG": False})
        assert again == r, (again, r)
        print("harness tests ok", {k: r[k] for k in ("mrr_a", "mrr_b", "delta", "leak_sensitive")})
        return 0
    finally:
        shutil.rmtree(td, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
