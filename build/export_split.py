# %% [markdown]
# # Export the FPNet train/unseen split as a small key list
#
# **Colab, CPU-only runtime, ~2 minutes, 0 compute units.**
#
# Stage C v2 needs to know which structures the FPNet actually gradient-updated on. That
# information already exists in `_fpnet/pack/splits.npz` from Stage A — this notebook just
# extracts the InChIKey14 lists so they can be attached to a Kaggle notebook without
# uploading the 1.9 GB fingerprint matrix.
#
# Three disjoint sets come out:
#
# | set | FPNet saw it? | used for |
# |---|---|---|
# | `seen_keys` | **yes**, trained on | Stage-C cells **A** and **C** — the Class-1 regime |
# | `es_keys` | **no** (scaffold-disjoint holdout) | Stage-C cells **D** and **B** — the Class-2 regime |
# | `banned_keys` | no (val `ban_keys`) | **reserved for Stage D** — never a Stage-C query |
#
# The split rule is taken straight from Stage A4: a structure was trained on iff
# `~struct_es & ~struct_banned`.

# %%
from pathlib import Path
import numpy as np

from google.colab import drive
drive.mount("/content/drive")

PACK = Path("/content/drive/MyDrive/casmi_2026/_fpnet/pack")
OUT = Path("/content/drive/MyDrive/casmi_2026/_fpnet/casmi26-fpnet-v3/fpnet_split_keys.npz")

uni = np.load(PACK / "uni.npz", allow_pickle=True)
sp = np.load(PACK / "splits.npz", allow_pickle=True)

ik = np.asarray(uni["ik"], dtype=object)
struct_es = np.asarray(sp["struct_es"], dtype=bool)
struct_banned = np.asarray(sp["struct_banned"], dtype=bool)
assert len(ik) == len(struct_es) == len(struct_banned), "splits.npz is not aligned with uni.npz"

# Stage A4: is_train = ~struct_es[yrow] & ~is_banned  ->  trained iff ~es & ~banned
trained = ~struct_es & ~struct_banned
unseen_es = struct_es & ~struct_banned

U = lambda a: np.asarray([str(x) for x in a], dtype="U14")   # fixed-width: no pickle needed
seen_keys = U(ik[trained])
es_keys = U(ik[unseen_es])
banned_keys = U(ik[struct_banned])

assert not (set(seen_keys) & set(es_keys)), "seen/es overlap"
assert not (set(seen_keys) & set(banned_keys)), "seen/banned overlap"

OUT.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(OUT, seen_keys=seen_keys, es_keys=es_keys, banned_keys=banned_keys)

print(f"structures total        : {len(ik):,}")
print(f"  seen    (FPNet trained): {len(seen_keys):,}")
print(f"  es      (never trained): {len(es_keys):,}   <- Stage-C Class-2 regime")
print(f"  banned  (val holdout)  : {len(banned_keys):,}   <- Stage D only")
print(f"\nwrote {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")
print("\nAdd this file to the casmi26-fpnet-v3 Kaggle dataset, next to the checkpoints.")

if len(es_keys) < 5000:
    print(f"\n!! only {len(es_keys):,} unseen structures. After Stage C's NP-library / adduct /")
    print("   mass filters this may not support ~700 queries. Stage C prints the surviving")
    print("   count and falls back to the val-banned scaffold relatives if needed.")
