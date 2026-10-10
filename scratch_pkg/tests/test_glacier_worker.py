"""resources/glacier_worker.py on a stub casmi26-glacier package (stub gl_runner + stub torch; no model, no GPU), driven
through sim.GlacierWorker exactly as the scheduler calls it: request TSV -> NPZ, error codes, 5 eV buckets, the
precursor cut, and a non-zero exit on gl_runner's fail-fast.

Real-model smoke (optional): set CASMI_GLACIER_PKG to an extracted casmi26-glacier dir whose PYTHONPATH needs are met
(torch + RDKit 2025.03 importable), e.g. CASMI_GLACIER_PKG=/tmp/glacier python scratch_pkg/tests/test_glacier_worker.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from casmi_scratch import chem, sim  # noqa: E402
from casmi_scratch.config import Config  # noqa: E402

STUB_GL = r'''
import json, os, sys
import numpy as np
COV = ("[M+H]+", "[M+Na]+")
_CANON = {}

class Log:
    def __call__(self, *a):
        print(*a, file=sys.stderr)

def setup_env(a, log):
    assert a.site == "none" and not a.any_rdkit, a
    return "2025.03.6"

def load_model(pkg, dev, log, ckpt=None):
    return "model"

def smiles_ok(smi):
    return (False, "rdkit parse") if smi in ("", "bad") else (True, "")

def instr_token(sp):
    return sp["instr"] if sp.get("instr") in ("Orbitrap", "QTOF") else "QTOF"

def ce_bucket(ce):
    ceb = float(np.round(ce / 5.0) * 5.0) if np.isfinite(ce) else np.nan
    return 5.0 if np.isfinite(ceb) and ceb < 5 else ceb

class Predictor:
    def __init__(self, model, dev, bs, log):
        self.bs, self.cache, self.n_pred, self.n_fail = bs, {}, 0, 0

    def predict(self, jobs, deadline):
        if os.environ.get("STUB_FAILFAST"):
            raise RuntimeError("first 64 predictions all failed -> abort")
        with open(os.environ["STUB_JOBS"], "w") as f:
            json.dump(sorted(jobs), f)
        for j in jobs:
            self.n_pred += 1
            if j[0] == "CNONE":
                self.cache[j] = None
                self.n_fail += 1
            elif j[0] == "CEMPTY":
                self.cache[j] = (np.array([400.0]), np.array([1.0]))
            else:
                self.cache[j] = (np.array([50.0, 100.0, 400.0]), np.array([0.5, 0.25, 1.0]))
        return True
'''

STUB_TORCH = '''
class _Cuda:
    @staticmethod
    def is_available():
        return False
cuda = _Cuda()
class _Cudnn:
    benchmark = True
class backends:
    cudnn = _Cudnn()
def use_deterministic_algorithms(*a, **k):
    pass
'''


def stub_pkg(d):
    pkg = Path(d) / "casmi26-glacier"
    (pkg / "torch").mkdir(parents=True)
    (pkg / "gl_runner.py").write_text(STUB_GL)
    (pkg / "torch" / "__init__.py").write_text(STUB_TORCH)
    return pkg


def requests():
    rows = [("C", "[M+H]+", 41.0), ("C", "[M+H]+", 38.0), ("bad", "[M+H]+", 40.0), ("C", "[M-H]-", 40.0),
            ("C", "[M+H]+", float("nan")), ("CNONE", "[M+H]+", 40.0), ("CEMPTY", "[M+H]+", 40.0),
            ("C", "[M+Na]+", 2.0)]
    req = pd.DataFrame({"key": [f"K{i:013d}" for i in range(len(rows))], "smiles": [r[0] for r in rows],
                        "adduct": [r[1] for r in rows], "precursor_mz": 200.0, "ce_ev": [r[2] for r in rows]})
    req["ce"] = req.ce_ev
    req["instrument"] = "QTOF"
    return req


def test_worker_protocol_on_stub():
    with tempfile.TemporaryDirectory() as d:
        pkg = stub_pkg(d)
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
        env["STUB_JOBS"] = str(Path(d) / "jobs.json")
        w = sim.GlacierWorker(sys.executable, pkg, env, Path(d) / "work", device="cpu", sparse_k=100, prec_margin=2.0)
        req = requests()
        spectra, ok, err, timing, _ = w(req, "t")
        assert list(ok) == [True, True, False, False, False, False, False, True], (ok, err)
        assert list(err) == ["", "", "rdkit_parse:rdkit parse", "adduct_unsupported:[M-H]-", "ce_nan",
                             "predict:none", "empty_after_prec_cut", ""], err
        mz, it = spectra[0]
        assert list(mz) == [50.0, 100.0] and list(it) == [1.0, 0.5]          # 400 > 200 + 2 cut, base renormalised
        assert len(spectra[6][0]) == 0 and len(spectra[2][0]) == 0
        jobs = sorted(tuple(j) for j in json.load(open(env["STUB_JOBS"])))
        assert jobs == sorted([("C", "[M+H]+", "QTOF", 40.0), ("C", "[M+Na]+", "QTOF", 5.0),
                               ("CEMPTY", "[M+H]+", "QTOF", 40.0), ("CNONE", "[M+H]+", "QTOF", 40.0)]), jobs
        assert timing["model"] == "glacier" and timing["n_predicted"] == 3 and timing["rdkit"] == "2025.03.6"
        assert sim.err_counts(err) == {"rdkit_parse": 1, "adduct_unsupported": 1, "ce_nan": 1, "predict": 1,
                                       "empty_after_prec_cut": 1}
        env["STUB_FAILFAST"] = "1"
        try:
            w(req, "ff")
            raise AssertionError("a gl_runner fail-fast must surface as a worker failure")
        except RuntimeError as e:
            assert "worker failed" in str(e) and "fail" in str(e)


def test_real_glacier_smoke():
    pkg = os.environ.get("CASMI_GLACIER_PKG")
    if not pkg:
        return                                                         # optional: needs the model + its env
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([os.path.join(pkg, "shim"), os.path.join(pkg, "src")] +
                                        [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p])
    with tempfile.TemporaryDirectory() as d:
        w = sim.GlacierWorker(sys.executable, pkg, env, d, device="cpu")
        out = sim.smoke(w, Config().with_(sim_model="glacier"))
        print("[GLACIER smoke]", out)
        assert out["n_peaks_40"] > 0


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]

if __name__ == "__main__":
    for f in TESTS:
        f()
    print(f"test_glacier_worker: {len(TESTS)} OK")
