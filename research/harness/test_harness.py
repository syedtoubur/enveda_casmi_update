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
        scorecard_tests()
        return 0
    finally:
        shutil.rmtree(td, ignore_errors=True)


def scorecard_tests() -> None:
    # W12: the family is the change's, not the base config's
    a = {"METRIC_FUSION": True, "ICE_FALLBACK": True, "FUSION_WINDOW": 40}
    assert harness.knob_diff(a, {**a, "REL_PROMOTE": 500}) == {"REL_PROMOTE": 500}
    # real replay on the synthetic dump: one cohort, the rank-only verdict and the leak split are filled in
    one = harness.cohort_score("T", {}, {"USE_ENG": False})
    assert one["n"] == 2 and one["n_leak_hit"] + 0 <= 2 and one["ci90"][0] <= one["delta"] <= one["ci90"][1]
    # composite: equal class weights over non-advisory cohorts; advisory cohorts never count
    coh = {"C2": dict(cls="C2", delta=0.02), "C2_hard": dict(cls="C2", delta=0.0), "C1_npx": dict(cls="C1", delta=0.01),
           "C2_np": dict(cls="C2", delta=0.5, advisory=True)}
    assert abs(harness.composite(coh, {}) - (0.01 + 0.01) / 2) < 1e-12
    assert abs(harness.composite(coh, {"C1": 3.0}) - (3 * 0.01 + 0.01) / 4) < 1e-12
    # scorecard with a fake per-cohort scorer: harm on a non-target cohort kills; advisory harm only flags
    rows = {"C2": (0.02, 0.01, 0.03, 0), "C1_npx": (-0.01, -0.02, -0.006, 0), "C2_np": (-0.05, -0.08, -0.02, 0)}
    fake = lambda c, a, b, post: dict(n=400, cls=harness.COHORT_CLASS[c], delta=rows[c][0], ci90=list(rows[c][1:3]),
                                      up=1, down=0, displaced=rows[c][3], delta_leak_hit=None, delta_other=None)
    r = harness.scorecard({}, {"K": 1}, "C2", cohorts=["C2", "C1_npx", "C2_np"], scorer=fake)
    assert r["verdict"] == "DEAD" and any("harm on C1_npx" in d for d in r["dead"]), r["dead"]
    assert not any("C2_np" in d for d in r["dead"]) and any("C2_np" in f for f in r["flags"])
    rows["C1_npx"] = (0.001, -0.003, 0.004, 0)
    r = harness.scorecard({}, {"K": 1}, "C2", cohorts=["C2", "C1_npx"], scorer=fake)
    assert r["verdict"] == "QUEUE" and r["schema"] == "scorecard-1" and r["n_target"] == 400
    print("scorecard tests ok")


if __name__ == "__main__":
    raise SystemExit(main())
