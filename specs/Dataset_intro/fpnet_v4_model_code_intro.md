# CASMI v4 Models & Codebase Guide: Architecture, Key Files, and Validation Split Mechanics

## 1. Executive Summary

This document provides a comprehensive, agent-ready reference for the contents, structure, and execution logic of the **CASMI 2026 v4 / v4b model engine**, extracted from `casmi-v4-models.zip` (Kaggle dataset: [`ahmedberatozer/casmi26-v4b-models`](https://www.kaggle.com/datasets/ahmedberatozer/casmi26-v4b-models)).

The archive packages:
1. **The `casmi` core engine**: Spectral processing, FPNet transformer architectures, MetFrag-lite fragmentation, analog search, and training data loaders.
2. **The `fe_v4` feature engineering subsystem**: Intrinsic chemistry descriptors, advanced fragmentation families (`frag2`, `fragnet`), derivation priors, and multi-model agreement views.
3. **Precomputed lookup assets (`fe_v4/res/`)**: Binary arrays and graphs allowing rapid offline feature calculation.

Additionally, this document answers the exact mechanism behind **Train/Validation splits** in this pipeline and provides code to recreate an honest, leak-free split for retraining FPNet.

---

## 2. Directory Structure Overview

When extracted, the archive contains:

```text
casmi-v4-models/
├── casmi/                      # Core neural models, inference engine, and training data handlers
│   ├── __init__.py
│   ├── build.py                # CSR peak cache builder & instrument classification
│   ├── chem.py                 # Chemical physics, adduct masses, formula parser
│   ├── derive.py               # Biosynthetic class-3 candidate mutator
│   ├── dreams_model.py         # DreaMS foundation model backbone & DreamsFP adapter
│   ├── engine.py               # Multi-channel evidence ranking engine
│   ├── fpnet.py                # FPNet PyTorch transformer & contrastive ranking loss
│   ├── fpnet6.py               # Engine v6 FPNet dispatch (FPNet + DreamsFP dual mode)
│   ├── frag.py                 # In-silico fragmentation & peak explanation (MetFrag-lite)
│   ├── graphs.py               # Molecular graph tokens for cross-encoder
│   ├── library.py              # Reference spectral library indexing and analog matching
│   ├── pool.py                 # Unified candidate pool (train + COCONUT) search index
│   ├── ranker.py               # LightGBM LambdaRank reranker with grouped CV
│   ├── spectra.py              # Numba kernels for entropy and cosine similarity
│   ├── structs.py              # Multiprocess SMILES processing & fingerprint unpacker
│   ├── traindata.py            # ★ Training data access and batch generator for FPNet/X-Enc
│   └── xenc.py                 # Cross-encoder transformer (peak tokens x atom tokens)
│
└── fe_v4/                      # SOTA v4b feature engineering pipeline
    ├── fams/                   # Specialized feature engineering families
    │   ├── __init__.py
    │   ├── analog_struct.py    # Candidate-to-analog structural alignment
    │   ├── derivation.py       # Derivation transformation priors (Class-3)
    │   ├── frag2.py            # Fragmentation 2.0 (cleavage coverage & depth)
    │   ├── fragnet.py          # Learned fragmentation scorer ("learned MetFrag")
    │   └── model_views.py      # Consensus & agreement between FPNet A/B and DreamsFP
    ├── res/                    # Precomputed binary caches (~390 MB)
    │   ├── analog_struct/      # Graph hash, fvec, and SMILES buffers
    │   ├── derivation/         # Parity fixtures & prior transformation tables
    │   ├── frag2/              # Substructure capability vectors & neutral loss tables
    │   └── fragnet/            # Learned fragment counts, feature matrices, atom/bond graphs
    ├── chemfeat.py             # Intrinsic structure records & Murcko scaffold keys
    ├── ctxlib.py               # QueryView context library & shard manager
    ├── derive6.py              # Class-3 candidate biosynthetic transformations
    ├── fragcaps.py             # Substructure capabilities for neutral loss diagnostics
    ├── hooks_v1.py             # Diagnostic hooks into engine evaluation
    └── v1engine.py             # Feature extraction wrapper integrating the v1 backbone
```

---

## 3. Key Files & Subsystem Analysis

### 3.1 Neural FPNet Models (`casmi/fpnet.py`, `casmi/fpnet6.py`)
- **Concept**: Predicts a molecular fingerprint vector $z$ directly from clean mass spectra. The score of candidate structure $f$ under predicted logits $z$ is the inner product $f \cdot z$ (Bayes log-likelihood up to a constant).
- **Architecture**:
  - `SinEmb`: Multi-scale sinusoidal scalar embeddings of m/z, neutral loss (precursor m/z - peak m/z), and collision energy.
  - Transformer encoder: Multi-head attention (`F.scaled_dot_product_attention`) over peak tokens + global context token (precursor m/z, adduct, collision energy, ionization mode).
  - Dual head: Fingerprint classification head (`2*d -> 2048 -> nbits`) and elemental composition head (`2*d -> 256 -> 10 elements`).
- **`casmi/fpnet6.py`**: Extends model loading with dynamic dispatch: loads standard `FPNet` or `DreamsFP` depending on checkpoint architecture tag `ck['arch']`.

### 3.2 Training Data Pipeline (`casmi/traindata.py`)
- **Role**: Houses the `Data` class used to train FPNet and the cross-encoder.
- **Negative Sampling**: Contrastive learning uses hard in-batch negatives: for each positive structure, draws $K=63$ decoys within $\pm 10\text{ ppm}$ of the target mass.
- **Data Augmentations**:
  - Multi-spectrum merging: Groups spectra by `(sid, ionization_mode)` to simulate test-time multi-collision energy queries.
  - Spectrum jitter: Random peak dropout ($[0\%, 30\%]$), intensity noise ($\exp(\mathcal{N}(0, 0.3))$), and ppm m/z jitter ($\mathcal{N}(0, 4\times 10^{-6})$).
  - Collision energy dropout: Randomly flags CE as unknown ($10\%$ rate).
- **Validation MRR**: Contains `window_mrr()`, evaluating top-25 MRR within a $\pm 10\text{ ppm}$ pool window using GPU-accelerated dot products (`fp @ z`).

### 3.3 Cache & Spectrum Building (`casmi/build.py`, `casmi/spectra.py`)
- **`build_spec_cache()`**: Reads raw `train.parquet`, performs Numba peak cleaning (drops peaks $> \text{precursor} + 2\text{ Da}$, relative intensity floor $0.1\%$, top-512 peaks, normalized base peak $= 1.0$), and produces:
  - `work/spec_meta.parquet`: One row per spectrum with `sid`, `adduct_ix`, `mode`, `precursor_mz`, `ce_mean`, `ce_n`, `lib`.
  - `work/spec_off.npy`, `work/spec_mz.npy`, `work/spec_it.npy`: Compact CSR offset arrays for zero-copy memory mapping.
- **`casmi/spectra.py`**: Numba-optimized entropy similarity, modified cosine similarity, and hybrid mass-shifted spectral matching.

### 3.4 Evidence Engine & Reranker (`casmi/engine.py`, `casmi/ranker.py`)
- **`casmi/engine.py`**: Combines 7 evidence channels into 94 tabular features per candidate:
  1. **L (Library)**: Visible library entropy match.
  2. **A (Analog)**: Mass-shifted spectral similarity propagated via Tanimoto.
  3. **M (Model)**: FPNet $f \cdot z$ score (single and merged spectra).
  4. **F (Fragmentation)**: Peak explanation ratio.
  5. **S (Subformula)**: Peak subformula explanation.
  6. **G (Generated)**: Biosynthetic Class-3 candidate mutation flags.
  7. **XS (Cross-encoder)**: Joint peak-atom transformer reranking score.
- **`casmi/ranker.py`**: LightGBM `lambdarank` wrapper (`RankerBag`) trained with structure-grouped cross-validation (`cv_report`) evaluating `mrr_at(pred, y, g, k=25)`.

### 3.5 Feature Engineering v4b Subsystem (`fe_v4/`)
- Advanced modules calculating deeper discriminative features:
  - `fams/fragnet.py`: Learned MetFrag replacement that maps bond cleavage types and element graphs into likelihood ratios.
  - `fams/frag2.py`: Depth-aware fragmentation coverage measuring bond cleavage precision.
  - `fams/derivation.py`: Derivation evidence measuring whether a candidate could originate from visible parents via known metabolic reactions.
  - `fams/model_views.py`: Ensembling agreement between models (FPNet A, FPNet B, DreamsFP D).

---

## 4. Train / Validation Split Mechanics

### 4.1 Does `casmi-v4-models.zip` Contain the Split Generation Code?
**No.** The zip archive does **not** contain the standalone generator script that created the splits. Instead, the code is designed to **consume** an existing split file: `work/split.parquet`.

### 4.2 How `casmi/traindata.py` Implements the Split
Inside [casmi/traindata.py](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/.scratch/casmi_v4/casmi/traindata.py#L11-L43):
```python
class Data:
    def __init__(self, device, K=63, ppm=10.0, seed=0,
                 holdout_folds=('np', 'nplib', 'syn', 'plusk', 'twin')):
        ...
        split = pd.read_parquet('work/split.parquet')
        st = pd.read_parquet('work/train_structs.parquet', columns=['sid', 'key'])
        ...
        fold = np.array(['train'] * len(st), dtype=object)
        fold[split.sid.values] = split.fold.values
        self.fold = fold
        is_hold = np.isin(fold, holdout_folds)
        ok = (~is_hold[self.sid]) & (self.sid2pid[self.sid] >= 0)
        self.tr = np.where(ok)[0]
```

### 4.3 Schema of `work/split.parquet`
| Column | Type | Description |
|---|---|---|
| `sid` | `int64` | Structure ID aligning with `work/train_structs.parquet` |
| `fold` | `string` | Split assignment label (e.g. `'train'`, `'np'`, `'nplib'`, `'syn'`, `'plusk'`, `'twin'`) |

### 4.4 Original Author's Folds
In the author's internal pipeline:
- `'np'`: `enveda-np-examples` (the competition's 250 Natural Product molecules on timsTOF).
- `'nplib'`: Natural product reference libraries (GNPS, RIKEN, MassBank, MoNA).
- `'plusk'`: Pluskal laboratory dataset (`pluskal_ms2`).
- `'syn'`: Synthetic library (`enveda-180`, `drug_plus`).
- `'twin'`: Murcko scaffold or 2D connectivity relatives of held-out query molecules.

### 4.5 Why `fpnet_full1` Outperforms `fpnet_0`
- **`fpnet_0`**: Trained with `holdout_folds=('np', 'nplib', 'syn', 'plusk', 'twin')`. All natural products and benchmark datasets were withheld.
- **`fpnet_full1`**: Trained on **all folds** (`holdout_folds=()`), including `enveda-np-examples` and all NP libraries. Because the competition test set consists of natural products, training on natural product examples provided a significant **+0.020 to +0.035 MRR jump**.

---

## 5. How to Recreate the Train/Validation Split for Retraining

If you need to train an FPNet model with the full capacity of `fpnet_full1` while keeping a legitimate validation set for early stopping and MRR tracking:

### 5.1 Using the Repository's Honest Validation Set (`val_scripts.zip`)
This repository contains `val_scripts.zip`, which defines an honest 1,145-molecule validation set (`val.parquet`, `val_key.pkl`, and `scaffolds.pkl`).

### 5.2 Split Generation Script

Run this script to generate `work/split.parquet`:

```python
"""Generate work/split.parquet using the official 1,145-molecule honest validation set."""
import os
import pickle
import numpy as np
import pandas as pd

WORK_DIR = "work"
os.makedirs(WORK_DIR, exist_ok=True)

# 1. Load train structures table
structs_path = os.path.join(WORK_DIR, "train_structs.parquet")
st = pd.read_parquet(structs_path)  # columns: sid, key, inchikey14

# 2. Load validation keys and scaffold lookup
with open("val_scripts/val_key.pkl", "rb") as f:
    val_keys = pickle.load(f)
val_inchikeys = set(item["inchikey14"] for item in val_keys)

with open("val_scripts/scaffolds.pkl", "rb") as f:
    scaffold_map = pickle.load(f)  # dict: inchikey14 -> scaffold_id

# Find scaffolds of the validation structures
val_scaffolds = set(scaffold_map.get(k) for k in val_inchikeys if k in scaffold_map)

# 3. Label each structure with its fold
folds = []
for _, row in st.iterrows():
    ik14 = row["inchikey14"]
    scaf = scaffold_map.get(ik14)
    if ik14 in val_inchikeys:
        folds.append("val")       # Primary validation holdout
    elif scaf in val_scaffolds:
        folds.append("twin")      # Scaffold twin (exclude from train to prevent leak)
    else:
        folds.append("train")     # Training fold

# 4. Save work/split.parquet
split_df = pd.DataFrame({
    "sid": st["sid"].values.astype(np.int64),
    "fold": np.asarray(folds, dtype=object)
})
split_df.to_parquet(os.path.join(WORK_DIR, "split.parquet"), index=False)

print(f"Generated work/split.parquet:")
print(split_df["fold"].value_counts())
```

### 5.3 Training Invocation
In your FPNet training loop:
1. Initialize the dataset with:
   ```python
   data = Data(device="cuda", K=63, holdout_folds=("val", "twin"))
   ```
2. Retrieve validation spectra IDs:
   ```python
   val_spec_ids = np.where(data.fold[data.sid] == "val")[0]
   ```
3. Evaluate checkpoint quality periodically:
   ```python
   mrr25, count = window_mrr(model, data, val_spec_ids, device="cuda", merged=True)
   print(f"Validation MRR@25: {mrr25:.4f} across {count} validation groups")
   ```
