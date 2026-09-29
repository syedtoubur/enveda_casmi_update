"""P0.1a official DreaMS leg: runs INSIDE the DreaMS venv (Python 3.11, upstream pins), never in the kernel.

Loads the embedding model the way `casmi_dreams_colab.ipynb` did (`PreTrainedModel.from_name`), then
  1. dumps the backbone args, the head hparams and the checkpoint SHA1s (A2, A3),
  2. converts it once to `dreams_embed.pt` with build/dreams_infer.py `save_dreams` (A2),
  3. embeds the parity spectra with the official `dreams_embeddings` over an MSData HDF5 (leg a),
     exactly as cell 8 built the cache,
  4. reloads `dreams_embed.pt` into the vendored network, strict, and embeds the same spectra (leg b,
     venv copy; the notebook recomputes leg b in its own interpreter).

Usage (from the notebook):
    <venv python> dreams_official_worker.py --infer-dir DIR --ckpt embedding_model.ckpt \
        --parity parity_in.npz --out-pt dreams_embed.pt --out-emb official.npz --out-info info.json \
        --commit <sha>
"""
import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path


def sha1(path, chunk=1 << 20):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def jsonable(v):
    try:
        json.dumps(v)
        return v
    except TypeError:
        return repr(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infer-dir", required=True)
    ap.add_argument("--ckpt", default="", help="embedding_model.ckpt cached on Drive ('' = fetch from HF)")
    ap.add_argument("--parity", required=True)
    ap.add_argument("--out-pt", required=True)
    ap.add_argument("--out-emb", required=True)
    ap.add_argument("--out-info", required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    a = ap.parse_args()

    sys.path.insert(0, a.infer_dir)
    import numpy as np
    import pandas as pd
    import torch
    import dreams_infer as di
    import dreams.utils.data as du
    from dreams.api import PreTrainedModel, dreams_embeddings
    from dreams.definitions import DREAMS_EMBEDDING, PRETRAINED

    PRETRAINED.mkdir(parents=True, exist_ok=True)
    tgt = PRETRAINED / "embedding_model.ckpt"
    if not tgt.exists() and a.ckpt:
        shutil.copy(a.ckpt, tgt)       # the copy that built the cache; otherwise from_name fetches HF's
    pm = PreTrainedModel.from_name(DREAMS_EMBEDDING)     # fetches ssl_model.ckpt from HF (A1)
    model = pm.model
    args = model.backbone.hparams["args"]
    args_d = di.args_to_dict(args)
    info = {
        "dreams_commit": a.commit,
        "python": sys.version.split()[0], "torch": torch.__version__,
        "embedding_model_sha1": sha1(tgt), "ssl_model_sha1": sha1(PRETRAINED / "ssl_model.ckpt"),
        "n_highest_peaks": pm.n_highest_peaks,
        "dtype": str(next(model.parameters()).dtype),
        "head_class": type(model).__name__,
        "head_hparams": {k: jsonable(v) for k, v in dict(model.hparams).items()},
        "backbone_args": args_d,
        "backbone_args_all": {k: jsonable(v) for k, v in vars(args).items()},
        "n_params": int(sum(p.numel() for p in model.parameters())),
    }
    assert args_d["charge_feature"] is False or args_d["charge_feature"] == 0, "charge_feature is set"
    assert pm.n_highest_peaks == di.N_PEAKS, pm.n_highest_peaks
    assert info["dtype"] == "torch.float32", info["dtype"]

    meta = {k: info[k] for k in ("dreams_commit", "embedding_model_sha1", "ssl_model_sha1", "torch")}
    di.save_dreams(a.out_pt, args, model.state_dict(), meta)
    info["dreams_embed_sha1"] = sha1(a.out_pt)
    info["state_dict_keys"] = len(di.inference_state_dict(model.state_dict()))

    z = np.load(a.parity)
    off, mz, it, prec = z["off"], z["mz"], z["it"], z["prec"]
    pls = [np.stack([mz[off[i]:off[i + 1]], it[off[i]:off[i + 1]]]).astype(np.float32)
           for i in range(len(prec))]

    # leg (a): the official path, as casmi_dreams_colab.ipynb cell 8
    t0 = time.time()
    h5 = Path(a.out_emb).with_suffix(".hdf5")
    h5.unlink(missing_ok=True)
    df = pd.DataFrame({"spectrum": pls, "precursor_mz": prec.astype(np.float64)})
    msdata = du.MSData.from_pandas(df, n_highest_peaks=di.N_PEAKS, hdf5_pth=h5, in_mem=False, mode="r")
    emb_a = np.asarray(dreams_embeddings(msdata, batch_size=a.batch_size, progress_bar=False), np.float32)
    info["t_official_s"] = time.time() - t0

    # leg (b, venv copy): the converted file in the vendored network, strict load
    t0 = time.time()
    vend, vmeta = di.load_dreams(a.out_pt, device="cpu", expect_sha1=info["dreams_embed_sha1"])
    emb_b = di.dreams_embed(vend, pls, prec, batch_size=a.batch_size)
    info["t_vendored_venv_s"] = time.time() - t0
    info["official_device"] = str(next(model.parameters()).device)

    np.savez(a.out_emb, emb_a=emb_a, emb_b_venv=emb_b)
    Path(a.out_info).write_text(json.dumps(info, indent=1, sort_keys=True))
    print(json.dumps({k: info[k] for k in ("dreams_embed_sha1", "embedding_model_sha1", "ssl_model_sha1",
                                           "n_highest_peaks", "head_class", "t_official_s")}))


if __name__ == "__main__":
    main()
