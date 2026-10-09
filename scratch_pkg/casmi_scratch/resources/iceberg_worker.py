"""ICEBERG forward-simulation worker for specs_simple_ms_simulator.

Runs INSIDE the isolated ms-pred environment (its own torch 2.6 / DGL / rdkit 2025.03), never in
the notebook's main interpreter -- that keeps RDKit 2026.03.3 (the metric's version) untouched.
The notebook talks to it through files only:

    request tsv      : key, smiles, adduct (CASMI spelling), precursor_mz, ce, instrument
                       (TSV, not parquet: the ms-pred environment ships no pyarrow)
    response npz     : off (n+1,) int64, mz / it float32 flat, ok bool, err str, timing json,
                       above_n int32 / above_share float32 / above_off float32 (per row, see below)

Peaks are the top `--sparse-k` by intensity, sorted by m/z, base peak = 1.0. m/z values are the
charged fragment masses (ICEBERG applies the adduct shift itself).

Peaks above precursor_mz + `--prec-margin` (2 Da) are dropped BEFORE the top-k, the same cleaning the
organisers applied to the test spectra. They exist because the MassSpecGym inten checkpoints have
`include_unshifted_mz=True`: every fragment is also emitted without the adduct shift, and for
water-loss adducts the unshifted root lands 17-35 Da above the precursor. What was dropped is
recorded per row: above_n = peak count, above_share = share of raw intensity, above_off = the
largest dropped m/z minus the precursor (NaN when nothing was dropped). A row left with no peaks
gets err "empty_after_prec_cut".

Reused unchanged by Phase 0 (gate), Phase 2 (precompute) and Phase 4 (Kaggle on-the-fly).

    python iceberg_worker.py --gen G.ckpt --inten I.ckpt --inp req.tsv --out resp.npz
    python iceberg_worker.py --gen G.ckpt --inten I.ckpt --hparams-only --out hp.json
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger

import importlib.abc  # noqa: E402
import importlib.machinery  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import types  # noqa: E402

STUBBED = []


class _Missing:
    """Stands in for a name from an ms_pred module the package does not ship; fails only if actually used."""

    def __init__(self, name):
        self._name = name

    def __call__(self, *a, **k):
        raise RuntimeError(f"{self._name} is not shipped with this ms_pred copy")

    def __getattr__(self, attr):
        raise RuntimeError(f"{self._name}.{attr} is not shipped with this ms_pred copy")


class _StubMissingMsPred(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """The casmi26-iceberg package ships a trimmed ms_pred (inference modules only) whose package __init__s still
    import training-side modules (e.g. nn_utils.tune_utils). Last on sys.meta_path, so it only answers for an
    ms_pred submodule no real finder can find, with an empty module."""

    def find_spec(self, name, path=None, target=None):
        return importlib.machinery.ModuleSpec(name, self) if name.startswith("ms_pred.") else None

    def create_module(self, spec):
        m = types.ModuleType(spec.name)
        m.__all__ = []

        def __getattr__(attr, _n=spec.name):
            if attr.startswith("__"):
                raise AttributeError(attr)
            return _Missing(f"{_n}.{attr}")
        m.__getattr__ = __getattr__
        return m

    def exec_module(self, module):
        STUBBED.append(module.__name__)


class _Anything:
    """Permissive placeholder served by an absent third-party module (import-time use only; ice_runner.py's)."""

    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        return _Anything()

    def __getattr__(self, k):
        if k.startswith("__"):
            raise AttributeError(k)
        return _Anything()

    def __getitem__(self, k):
        return _Anything()

    def __setitem__(self, k, v):
        pass

    def __iter__(self):
        return iter(())

    def __bool__(self):
        return False


class _StubModule(types.ModuleType):
    def __getattr__(self, k):
        if k.startswith("__"):
            raise AttributeError(k)
        v = type(k, (_Anything,), {}) if k[:1].isupper() else _Anything()
        setattr(self, k, v)
        return v


class _StubThirdParty(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Serves stub modules (and any submodule) for third-party roots that ms_pred imports at module level but never
    uses on the predict_mol path, when the image lacks them (multiprocess, h5py, pygmtools on the cp313 image)."""

    def __init__(self, roots):
        self.roots = set(roots)

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] not in self.roots:
            return None
        return importlib.machinery.ModuleSpec(name, self, is_package=True)

    def create_module(self, spec):
        m = _StubModule(spec.name)
        m.__path__ = []
        return m

    def exec_module(self, module):
        pass


# the casmi26-iceberg ice_runner.py list: imported at ms_pred module level, unused by ICEBERG inference
OPTIONAL = ("pygmtools", "h5py", "multiprocess", "platformdirs")

if os.environ.get("CASMI_ICE_PKG") == "1":
    sys.meta_path.append(_StubMissingMsPred())
    _absent = []
    for _m in OPTIONAL:
        try:
            importlib.import_module(_m)
        except BaseException:
            for _k in [k for k in sys.modules if k.split(".")[0] == _m]:
                del sys.modules[_k]
            _absent.append(_m)
    if _absent:
        sys.meta_path.insert(0, _StubThirdParty(_absent))
        print(f"[worker] absent third-party modules, stubbed: {_absent}", file=sys.stderr, flush=True)

import ms_pred.common as common  # noqa: E402
import ms_pred.common.chem_utils as _chem_utils  # noqa: E402
from ms_pred.iceberg import gen_model, inten_model, joint_model  # noqa: E402

if STUBBED:
    print(f"[worker] ms_pred modules not shipped, stubbed: {STUBBED}", file=sys.stderr, flush=True)

# predict_mol's InChI round trip writes a constant tautomer-transform file under platformdirs' user_cache_dir:
# point it at a writable temp dir (also needed when platformdirs itself is stubbed)
import tempfile  # noqa: E402
_CACHE = os.path.join(tempfile.gettempdir(), "ice_ms_pred_cache")
_chem_utils.user_cache_dir = lambda name, *a, **k: os.path.join(_CACHE, name)

import rdkit  # noqa: E402
print(f"[worker] rdkit {rdkit.__version__} at {os.path.dirname(rdkit.__file__)} | torch {torch.__version__}",
      file=sys.stderr, flush=True)
if os.environ.get("CASMI_ICE_PKG") == "1" and not rdkit.__version__.startswith("2025.03"):
    raise RuntimeError(f"rdkit {rdkit.__version__}: the casmi26-iceberg checkpoints need 2025.03 "
                       "(2026.03 kekulizes differently) -- the ice site is not first on PYTHONPATH")

RDLogger.DisableLog("rdApp.*")

# CASMI adduct spelling -> ms-pred spelling (only where they differ)
ADDUCT_MAP = {"[M-2H2O+H]+": "[M+H-2H2O]+", "[M+CH2O2-H]-": "[M+CHO2]-"}


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gen", required=True)
    p.add_argument("--inten", required=True)
    p.add_argument("--inp")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-nodes", type=int, default=100)
    p.add_argument("--sparse-k", type=int, default=100)
    p.add_argument("--threshold", type=float, default=0.0)
    p.add_argument("--prec-margin", type=float, default=2.0)
    p.add_argument("--hparams-only", action="store_true")
    return p.parse_args()


def _jsonable(d):
    out = {}
    for k, v in dict(d).items():
        try:
            json.dumps(v)
            out[k] = v
        except TypeError:
            out[k] = str(v)
    return out


# PyTorch 2.6 defaults weights_only=True, which breaks legacy Lightning checkpoints containing PosixPath/hparams.
try:
    from pathlib import PosixPath, WindowsPath
    torch.serialization.add_safe_globals([PosixPath, WindowsPath])
except Exception:
    pass

_orig_torch_load = torch.load
def _safe_torch_load(*load_args, **load_kwargs):
    if "weights_only" not in load_kwargs:
        load_kwargs["weights_only"] = False
    return _orig_torch_load(*load_args, **load_kwargs)
torch.load = _safe_torch_load


HP_KEYS = ("hyper_parameters", "hparams", "hp", "config", "kwargs", "args")
SD_KEYS = ("state_dict", "model_state_dict", "model", "weights")


def load_model(cls, path):
    """A Lightning checkpoint (.ckpt), or a plain torch file (.pt, e.g. the casmi26-iceberg package): a pickled
    module, or a dict holding hyper-parameters + state_dict."""
    if str(path).endswith(".pt"):   # the casmi26-iceberg layout: {hyper_parameters, state_dict}, as ice_runner.py loads it
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict) and "hyper_parameters" in obj and "state_dict" in obj:
            m = cls(**obj["hyper_parameters"])
            m.load_state_dict(obj["state_dict"], strict=True)
            return m
    try:
        try:
            return cls.load_from_checkpoint(path, map_location="cpu", weights_only=False)
        except TypeError:
            return cls.load_from_checkpoint(path, map_location="cpu")
    except Exception as e:
        lightning_err = e
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, torch.nn.Module):
        return obj
    if isinstance(obj, dict):
        hp = next((obj[k] for k in HP_KEYS if k in obj), None)
        sd = next((obj[k] for k in SD_KEYS if k in obj), None)
        if isinstance(sd, torch.nn.Module):
            return sd
        if hp is not None and sd is not None:
            m = cls(**dict(vars(hp) if hasattr(hp, "__dict__") and not isinstance(hp, dict) else hp))
            m.load_state_dict(sd)
            return m
    raise RuntimeError(f"cannot load {path} as {cls.__name__}: Lightning said {lightning_err!r}; file holds "
                       f"{sorted(obj)[:30] if isinstance(obj, dict) else type(obj)}")


def load(args):
    return load_model(gen_model.FragGNN, args.gen), load_model(inten_model.IntenGNN, args.inten)


def to_peaks(s, k, prec, margin):
    """(n, 2) array [mz, inten] -> (mz, it, above_n, above_share, above_off).

    Drops peaks above prec + margin, then keeps the top-k by intensity, sorted by m/z, base = 1.
    """
    s = np.asarray(s, dtype=np.float64).reshape(-1, 2)
    s = s[np.isfinite(s).all(axis=1) & (s[:, 1] > 0)]
    above = s[:, 0] > prec + margin
    above_n = int(above.sum())
    above_share = float(s[above, 1].sum() / s[:, 1].sum()) if len(s) else 0.0
    above_off = float(s[above, 0].max() - prec) if above_n else np.nan
    s = s[~above]
    if len(s) == 0:
        return np.empty(0, np.float32), np.empty(0, np.float32), above_n, above_share, above_off
    s = s[np.argsort(s[:, 1])[::-1][:k]]
    s = s[np.argsort(s[:, 0])]
    return (s[:, 0].astype(np.float32), (s[:, 1] / s[:, 1].max()).astype(np.float32),
            above_n, above_share, above_off)


def predict_batch(model, rows, args):
    return model.predict_mol(
        [r["smiles"] for r in rows],
        collision_eng=[float(r["ce"]) for r in rows],
        precursor_mz=[float(r["precursor_mz"]) for r in rows],
        adduct=[r["adduct"] for r in rows],
        instrument=[r["instrument"] for r in rows],
        threshold=args.threshold,
        device=args.device,
        max_nodes=args.max_nodes,
        adduct_shift=False,
        canonical_root_smi=True,
        binned_out=False,
    )["spec"]


def main():
    args = get_args()
    t0 = time.time()
    gen, inten = load(args)

    if args.hparams_only:
        with open(args.out, "w") as f:
            json.dump({"gen": _jsonable(gen.hparams), "inten": _jsonable(inten.hparams),
                       "ion2onehot_pos": common.ion2onehot_pos,
                       "instrument2onehot_pos": common.instrument2onehot_pos}, f, indent=1)
        return

    model = joint_model.JointModel(gen_model_obj=gen, inten_model_obj=inten).to(args.device)
    model.eval()
    model.freeze()
    t_load = time.time() - t0

    df = pd.read_csv(args.inp, sep="	")
    n = len(df)
    peaks = [None] * n
    err = [""] * n
    above_n = np.zeros(n, np.int32)
    above_share = np.zeros(n, np.float32)
    above_off = np.full(n, np.nan, np.float32)

    # canonicalise exactly as predict_smis.py does; reject what ms-pred cannot take
    rows = []
    for i, r in enumerate(df.to_dict("records")):
        m = Chem.MolFromSmiles(r["smiles"]) if isinstance(r["smiles"], str) else None
        ad = ADDUCT_MAP.get(r["adduct"], r["adduct"])
        if m is None:
            err[i] = "rdkit_parse"
        elif ad not in common.ion2onehot_pos:
            err[i] = f"adduct_unsupported:{r['adduct']}"
        elif r["instrument"] not in common.instrument2onehot_pos:
            err[i] = f"instrument_unsupported:{r['instrument']}"
        elif not np.isfinite(r["ce"]):
            err[i] = "ce_nan"
        else:
            rows.append(dict(r, i=i, smiles=Chem.MolToSmiles(m), adduct=ad))

    t1 = time.time()
    with torch.no_grad():
        for b in range(0, len(rows), args.batch_size):
            batch = rows[b:b + args.batch_size]
            try:
                specs = predict_batch(model, batch, args)
            except Exception:
                # isolate the failing molecule(s); keep the rest of the batch
                specs = []
                for r in batch:
                    try:
                        specs.append(predict_batch(model, [r], args)[0])
                    except Exception as e:  # noqa: BLE001
                        specs.append(None)
                        err[r["i"]] = f"predict:{type(e).__name__}:{str(e)[:120]}"
            for r, s in zip(batch, specs):
                if s is not None:
                    i = r["i"]
                    mz_, it_, above_n[i], above_share[i], above_off[i] = to_peaks(
                        s.detach().cpu().numpy(), args.sparse_k, float(r["precursor_mz"]), args.prec_margin)
                    peaks[i] = (mz_, it_)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    t_pred = time.time() - t1

    empty = (np.empty(0, np.float32), np.empty(0, np.float32))
    ok = np.array([p is not None and len(p[0]) > 0 for p in peaks])
    for i in range(n):
        if peaks[i] is not None and len(peaks[i][0]) == 0 and not err[i]:
            err[i] = "empty_after_prec_cut" if above_n[i] else "empty_spectrum"
    peaks = [p if p is not None else empty for p in peaks]
    lens = np.array([len(p[0]) for p in peaks], dtype=np.int64)
    off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    mz = np.concatenate([p[0] for p in peaks]) if n else np.empty(0, np.float32)
    it = np.concatenate([p[1] for p in peaks]) if n else np.empty(0, np.float32)
    timing = dict(n=n, n_predicted=int(ok.sum()), t_load=t_load, t_pred=t_pred,
                  mol_per_s=float(len(rows) / t_pred) if t_pred > 0 else None,
                  device=args.device, batch_size=args.batch_size, prec_margin=args.prec_margin,
                  gpu=torch.cuda.get_device_name(0) if args.device.startswith("cuda") else "cpu")
    np.savez(args.out, off=off, mz=mz, it=it, ok=ok, err=np.array(err, dtype="U200"),
             above_n=above_n, above_share=above_share, above_off=above_off, timing=json.dumps(timing))
    print(json.dumps(timing))


if __name__ == "__main__":
    main()
