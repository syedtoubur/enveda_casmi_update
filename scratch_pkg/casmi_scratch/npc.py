"""NPClassifier labels for the pool (R2): fingerprint, model loading, top-k decoding, pool_npc.parquet I/O.

NPClassifier = Kim et al., J. Nat. Prod. 2021, github.com/mwang87/NP-Classifier (MIT). Weights: Zenodo 5068687
(`model.zip` -> NP_classifier_{pathway,superclass,class}_V1.hdf5, Keras / TF 2.3). Label names:
`Classifier/dict/index_v1.json` ({"Pathway"|"Superclass"|"Class": {name: output index}}). Inputs, as the repo's
`fingerprint_handler.calculate_fingerprint(smiles, 2)`: `input_2048` = radius-0 counted Morgan bits, `input_4096` =
radius-1 and radius-2 counted bits (explicit H, 2048 bits each, only environments of exactly that radius).
Labels are computed once on Colab; Kaggle only reads pool_npc.parquet (no TensorFlow there).

The V1 weights were trained on the Morgan bits of RDKit <= 2021.03; RDKit 2021.09 changed them, and on today's bits
the net outputs noise (R2.0 FAIL, gate_report). Real labelling therefore takes its fingerprints from
`fp_subprocess` (resources/npc_fp_worker.py under rdkit-pypi 2021.3.5). `predict` without an explicit fp_fn refuses a
newer RDKit; tests and the LR fallback (trained and applied on the same bits) call `fingerprints` directly.
"""
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HEADS = ("pathway", "superclass", "class")
INDEX_KEY = dict(pathway="Pathway", superclass="Superclass", **{"class": "Class"})
NPC_FILE = "pool_npc.parquet"
NSC = 3                                  # superclasses kept per structure
FP_RDKIT = "2021.3.5"                    # rdkit-pypi release with the Morgan bits the V1 weights were trained on
FP_PYTHON = "3.9"                        # newest CPython with rdkit-pypi 2021.3.5 wheels
FP_WORKER = Path(__file__).resolve().parent / "resources" / "npc_fp_worker.py"


def fingerprint(smiles, radi=2):
    """-> (formula (2048,), binary (2048*radi,)) int32 counts, or None if RDKit cannot parse.

    Same arithmetic as NP-Classifier's calculate_fingerprint: bit i of radius r counts the environments of exactly
    radius r that set it.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdMolDescriptors
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    mol = Chem.AddHs(mol)
    formula, binary = np.zeros(2048, np.int32), np.zeros(2048 * radi, np.int32)
    for r in range(radi + 1):
        bi = {}
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=r, bitInfo=bi, nBits=2048)
        for i in fp.GetOnBits():
            n = sum(1 for e in bi[i] if e[1] == r)
            if n == 0:
                continue
            if r == 0:
                formula[i] = n
            else:
                binary[2048 * (r - 1) + i] = n
    return formula, binary


def fingerprints(smiles):
    """List of SMILES -> (F (n, 2048), B (n, 4096), ok (n,)); rows RDKit cannot parse are zero and ok=False."""
    n = len(smiles)
    F, B, ok = np.zeros((n, 2048), np.float32), np.zeros((n, 4096), np.float32), np.zeros(n, bool)
    for j, s in enumerate(smiles):
        fp = fingerprint(s)
        if fp is not None:
            F[j], B[j], ok[j] = fp[0], fp[1], True
    return F, B, ok


def label_names(index_json):
    """index_v1.json -> {head: [name per output index]} (an index named twice keeps its first name)."""
    d = json.load(open(index_json)) if isinstance(index_json, (str, os.PathLike)) else index_json
    out = {}
    for head in HEADS:
        m = d[INDEX_KEY[head]]
        names = [None] * (max(m.values()) + 1)
        for name, i in sorted(m.items(), key=lambda x: (x[1], x[0])):
            if names[i] is None:
                names[i] = name
        out[head] = names
    return out


def load_models(model_dir):
    """{head: Keras model} from the three NP_classifier_*_V1.hdf5 files (tf_keras first: they are Keras 2 files)."""
    try:
        import tf_keras as K
    except ImportError:
        from tensorflow import keras as K
    files = dict(pathway="NP_classifier_pathway_V1.hdf5", superclass="NP_classifier_superclass_V1.hdf5",
                 **{"class": "NP_classifier_class_V1.hdf5"})
    return {h: K.models.load_model(os.path.join(model_dir, f), compile=False) for h, f in files.items()}


def model_inputs(model, F, B):
    """Feed the two fingerprints in the model's own input order (matched on width)."""
    by_w = {2048: F, 4096: B}
    return [by_w[int(t.shape[-1])] for t in model.inputs]


def topk(P, names, k):
    """P (n, m) probabilities -> (labels (n, k) object, probs (n, k) float32), best first."""
    P = np.asarray(P, np.float32)
    k = min(k, P.shape[1])
    idx = np.argsort(-P, axis=1, kind="stable")[:, :k]
    lab = np.asarray(names, dtype=object)[idx]
    return lab, np.take_along_axis(P, idx, axis=1)


def decode(preds, names, ok):
    """{head: P} -> label frame: sc1..3 / sp1..3 (superclass top 3), pathway / pathway_p, class / class_p."""
    sc, sp = topk(preds["superclass"], names["superclass"], NSC)
    out = {f"sc{i + 1}": sc[:, i] for i in range(sc.shape[1])}
    out.update({f"sp{i + 1}": sp[:, i] for i in range(sp.shape[1])})
    for head in ("pathway", "class"):
        lab, p = topk(preds[head], names[head], 1)
        out[head], out[f"{head}_p"] = lab[:, 0], p[:, 0]
    df = pd.DataFrame(out)
    df.loc[~ok, [c for c in df.columns if not c.endswith("_p") and not c.startswith("sp")]] = None
    df.loc[~ok, [c for c in df.columns if c.endswith("_p") or c.startswith("sp")]] = np.nan
    return df


def rdkit_bits_ok(version=None):
    """True iff this RDKit gives the Morgan bits NP-Classifier V1 was trained on (<= 2021.03)."""
    if version is None:
        import rdkit
        version = rdkit.__version__
    return tuple(int(x) for x in version.split(".")[:2]) <= (2021, 3)


def local_fingerprints_checked(smiles):
    """In-process fingerprints, only on an RDKit with the old bits (else the labels are noise)."""
    if not rdkit_bits_ok():
        import rdkit
        raise RuntimeError(f"RDKit {rdkit.__version__} has post-2021.09 Morgan bits; NP-Classifier needs "
                           f"fp_subprocess (rdkit-pypi {FP_RDKIT}) or an explicit fp_fn")
    return fingerprints(smiles)


def fp_env_cmd(python=FP_PYTHON):
    """Command prefix that runs the fingerprint worker in a uv-managed Python with rdkit-pypi FP_RDKIT.
    pandas / tqdm only for the repo's fingerprint_handler (parity check)."""
    return ["uv", "run", "--no-project", "--python", python, "--with", f"rdkit-pypi=={FP_RDKIT}", "--with", "numpy<2",
            "--with", "pandas", "--with", "tqdm", "python", "-I", str(FP_WORKER)]


def fp_subprocess(smiles, cmd=None, workers=None, ref_dir=None, workdir=None):
    """SMILES -> (F (n, 2048) f32, B (n, 4096) f32, ok (n,), meta {rdkit, parity}) from the worker in its own env."""
    cmd = list(cmd or fp_env_cmd())
    smiles = ["" if not isinstance(x, str) else x.replace("\n", "") for x in smiles]
    with tempfile.TemporaryDirectory(dir=workdir) as d:
        inp, out = os.path.join(d, "in.txt"), os.path.join(d, "out.npz")
        with open(inp, "w", encoding="utf-8") as fh:
            fh.write("\n".join(smiles) + "\n")
        args = cmd + ["--inp", inp, "--out", out, "--workers", str(workers or os.cpu_count() or 1)]
        if ref_dir:
            args += ["--ref", str(ref_dir)]
        r = subprocess.run(args, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(out):
            raise RuntimeError(f"npc_fp_worker failed ({r.returncode}):\n{r.stdout[-2000:]}\n{r.stderr[-3000:]}")
        with np.load(out) as z:
            assert len(z["ok"]) == len(smiles), (len(z["ok"]), len(smiles))
            return (z["F"].astype(np.float32), z["B"].astype(np.float32), z["ok"].astype(bool),
                    dict(rdkit=str(z["rdkit"]), parity=int(z["parity"])))


def fp_fn_subprocess(**kw):
    """fp_fn for predict / label_all: fingerprints from the old-RDKit worker."""
    return lambda smiles: fp_subprocess(smiles, **kw)[:3]


def predict(smiles, models, names, fp_fn=None, batch=4096):
    """SMILES -> decoded label frame (one row per SMILES, in order). fp_fn(smiles) -> (F, B, ok); default = in-process
    fingerprints, which refuse a new RDKit (pass fp_fn_subprocess() for real labels)."""
    F, B, ok = (fp_fn or local_fingerprints_checked)(list(smiles))
    F, B = np.asarray(F, np.float32), np.asarray(B, np.float32)
    preds = {h: m.predict(model_inputs(m, F, B), batch_size=batch, verbose=0) for h, m in models.items()}
    return decode(preds, names, np.asarray(ok, bool))


def label_all(smiles, models, names, ckpt_dir, fp_fn=None, chunk=20_000, log=print):
    """Every SMILES -> decoded label frame, one checkpoint parquet per chunk (a rerun resumes)."""
    from .pool_build import run_chunked
    cols = [f"sc{i + 1}" for i in range(NSC)] + [f"sp{i + 1}" for i in range(NSC)] + \
        ["pathway", "pathway_p", "class", "class_p"]

    def fn(s):
        return list(predict(s, models, names, fp_fn)[cols].itertuples(index=False, name=None))
    return run_chunked(list(smiles), fn, ckpt_dir, "npc", cols, chunk, workers=1, log=log)


# ---------------------------------------------------------------- pool_npc.parquet

def write_npc(pool_dir, labels):
    """labels: frame aligned row-for-row with pool_key.npy (mass order) -> pool_npc.parquet (with a key column)."""
    keys = np.load(os.path.join(pool_dir, "pool_key.npy")).astype(str)
    assert len(labels) == len(keys), (len(labels), len(keys))
    df = labels.reset_index(drop=True).copy()
    df.insert(0, "key", keys)
    for c in df.columns:
        if c.startswith("sp") or c.endswith("_p"):
            df[c] = df[c].astype(np.float32)
    path = os.path.join(pool_dir, NPC_FILE)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path + ".tmp")
    os.replace(path + ".tmp", path)
    return path


def load_npc(pool_dir):
    """pool_npc.parquet -> (superclass codes (n, 3) int16 with -1 = none, probs (n, 3) float32, names list),
    aligned with pool_key.npy. The file's key column must match pool_key.npy exactly."""
    t = pq.read_table(os.path.join(pool_dir, NPC_FILE)).to_pandas()
    keys = np.load(os.path.join(pool_dir, "pool_key.npy")).astype(str)
    assert len(t) == len(keys) and np.array_equal(t.key.to_numpy().astype(str), keys), "pool_npc misaligned"
    cols = [c for c in (f"sc{i + 1}" for i in range(NSC)) if c in t]
    names = sorted({v for c in cols for v in t[c].dropna().unique()})
    code = {n: i for i, n in enumerate(names)}
    sc = np.full((len(t), len(cols)), -1, np.int16)
    for j, c in enumerate(cols):
        sc[:, j] = t[c].map(code).fillna(-1).to_numpy(np.int16)
    sp = np.nan_to_num(t[[f"sp{i + 1}" for i in range(len(cols))]].to_numpy(np.float32), nan=0.0)
    return sc, sp, names


# ---------------------------------------------------------------- fallback (R2.0 FAIL): linear model on table labels

def lr_fit(smiles, labels, max_iter=200, seed=0):
    """Multinomial logistic regression on the NPClassifier fingerprint (both blocks, log1p counts)."""
    from sklearn.linear_model import SGDClassifier
    F, B, ok = fingerprints(list(smiles))
    X = np.log1p(np.hstack([F, B]))[ok]
    y = np.asarray(labels, dtype=object)[ok]
    clf = SGDClassifier(loss="log_loss", max_iter=max_iter, random_state=seed, n_jobs=-1)
    clf.fit(X, y)
    return clf


def lr_predict(clf, smiles):
    """-> label frame with sc1..3 / sp1..3 like `decode` (pathway / class not available)."""
    F, B, ok = fingerprints(list(smiles))
    P = clf.predict_proba(np.log1p(np.hstack([F, B])))
    sc, sp = topk(P, list(clf.classes_), NSC)
    df = pd.DataFrame({**{f"sc{i + 1}": sc[:, i] for i in range(sc.shape[1])},
                       **{f"sp{i + 1}": sp[:, i] for i in range(sp.shape[1])}})
    df.loc[~ok, [f"sc{i + 1}" for i in range(sc.shape[1])]] = None
    df.loc[~ok, [f"sp{i + 1}" for i in range(sp.shape[1])]] = np.nan
    return df
