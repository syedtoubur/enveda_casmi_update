# CASMI 2026: Master Dataset & Model Catalog

> **Reference File**: [`specs/Dataset_intro/dataset_catalog.json`](file:///e:/ENVEDA_casmi/specs/Dataset_intro/dataset_catalog.json)  
> **Purpose**: Single authoritative registry of all Kaggle dataset slugs, purpose descriptions, file manifests, and `kernel-metadata.json` attachment presets.

---

## 1. Quick Presets for `kernel-metadata.json`

### 🚀 SOTA 0.433 / Sovereign Zenith Lineage
To run [`casmi-notebook-0-433.ipynb`](file:///e:/ENVEDA_casmi/casmi-notebook-0-433.ipynb) or Sovereign Zenith without missing file exceptions, copy this exact `dataset_sources` array:

```json
{
  "competition_sources": [
    "enveda-casmi-2026"
  ],
  "dataset_sources": [
    "metric/rdkit-2026-3-3-wheel",
    "eveliaveldrine/rdkit-wheels",
    "ahmedberatozer/casmi26-v4b-models",
    "ahmedberatozer/casmi26-v3-models",
    "ahmedberatozer/casmi26-fpnet-full1",
    "ahmedberatozer/casmi26-v2-pool",
    "ahmedberatozer/casmi26-pubchem-tier",
    "dmitriigluzdov/casmi26-pubchem-popularity-prior",
    "ahmedberatozer/casmi26-iceberg",
    "ahmedberatozer/casmi26-glacier",
    "megayak/casmi26-simulated-ranker-rows",
    "prvsiyan/casmi26-fp-models-v2",
    "prvsiyan/casmi26-ranker-features",
    "prvsiyan/coconut-casmi26-candidates",
    "prvsiyan/chebi-lipidmaps-casmi26",
    "hengck23/hengck23-dreams-enveda-casmi26"
  ]
}
```

---

### 🧪 Specs Scratch Line (R1 Unified Pool, R2 Class Vote, R3 Probes)
For experiments using the unified pool:

```json
{
  "competition_sources": [
    "enveda-casmi-2026"
  ],
  "dataset_sources": [
    "metric/rdkit-2026-3-3-wheel",
    "eveliaveldrine/rdkit-wheels",
    "syedtoubur/casmi26-scratch-pool",
    "ahmedberatozer/casmi26-fpnet-full1",
    "ahmedberatozer/casmi26-glacier",
    "ahmedberatozer/casmi26-iceberg"
  ]
}
```

---

## 2. Complete Dataset Master Inventory

| Dataset Slug | Category | Kaggle URL | Key Files | What It Does / Why It Matters |
| :--- | :--- | :--- | :--- | :--- |
| **`metric/rdkit-2026-3-3-wheel`** | Python Wheel | [Kaggle](https://www.kaggle.com/datasets/metric/rdkit-2026-3-3-wheel) | `rdkit-2026.3.3-cp310/311/312/313.whl` | **Mandatory**: The official competition metric evaluation RDKit wheel. |
| **`eveliaveldrine/rdkit-wheels`** | Python Wheel | [Kaggle](https://www.kaggle.com/datasets/eveliaveldrine/rdkit-wheels) | `rdkit-2025.3.6-cp313*.whl`, `numpy`, `pillow` | **Critical Fix**: ICEBERG was built for cp312; this provides the cp313 build of 2025.3.6 so ICEBERG/GLACIER subprocesses run without crashing on Kaggle. |
| **`ahmedberatozer/casmi26-v4b-models`** | Models & Engine | [Kaggle](https://www.kaggle.com/datasets/ahmedberatozer/casmi26-v4b-models) | `fe_A.pt`, `fe_B.pt`, `ranker_v4b.pkl`, `MANIFEST.json`, `casmi/engine.py` | SOTA v4b engine code, feature engineering tables, and LightGBM ranker. |
| **`ahmedberatozer/casmi26-v3-models`** | Models & Engine | [Kaggle](https://www.kaggle.com/datasets/ahmedberatozer/casmi26-v3-models) | `fpnet_0.pt`, `fpnet_1.pt`, `ranker_0.pkl`, `casmi/engine.py` | FPNet models used in the isolated PubChem background worker (`pc_runner.py`). |
| **`ahmedberatozer/casmi26-fpnet-full1`** | Model Weight | [Kaggle](https://www.kaggle.com/datasets/ahmedberatozer/casmi26-fpnet-full1) | `fpnet_full1.pt` | Drop-in FPNet trained on all competition folds + NP examples (+0.02 to +0.035 MRR jump). |
| **`prvsiyan/casmi26-fp-models-v2`** | Model Weight | [Kaggle](https://www.kaggle.com/datasets/prvsiyan/casmi26-fp-models-v2) | `fp_merged_m1.pt`, `fp_single_s2.pt` | Dual-view FPNet models used in Engine 2 analog propagation. |
| **`ahmedberatozer/casmi26-pubchem-tier`** | Candidate Pool | [Kaggle](https://www.kaggle.com/datasets/ahmedberatozer/casmi26-pubchem-tier) | `pc_mass.npy`, `pc_off.npy`, `pc_smiles.npy` | 105.88M PubChem structures memory-mapped on disk for zero-RAM sub-millisecond retrieval. |
| **`ahmedberatozer/casmi26-v2-pool`** | Candidate Pool | [Kaggle](https://www.kaggle.com/datasets/ahmedberatozer/casmi26-v2-pool) | `pool_meta.parquet`, `pool_fp.npy`, `pool_frag_*.npy` | Primary candidate pool (710,701 structures: Train + COCONUT 2.0). |
| **`syedtoubur/casmi26-scratch-pool`** | Candidate Pool | [Kaggle](https://www.kaggle.com/datasets/syedtoubur/casmi26-scratch-pool) | `r1_pool_meta.parquet`, `r1_pool_fp.npy` | User's unified candidate pool for the specs_scratch line (R1 pool). |
| **`cosmicbe/casmi26-pubchemlite`** | Candidate Pool | [Kaggle](https://www.kaggle.com/datasets/cosmicbe/casmi26-pubchemlite) | `pubchemlite_fp6930.npy`, `pubchemlite_meta.parquet` | 385k candidate artifact with 6,930-bit packed fingerprints for Engine 2 candidate expansion. |
| **`dmitriigluzdov/casmi26-pubchem-popularity-prior`** | Prior / Knowledge | [Kaggle](https://www.kaggle.com/datasets/dmitriigluzdov/casmi26-pubchem-popularity-prior) | `pc_lsid.npy`, `pc_lpmid.npy`, `pc_np.npy`, `pool_lsid.npy` | PubChem deposit & PubMed citation counts + NP flags (+0.012 to +0.016 LB boost). |
| **`dmitriigluzdov/casmi26-natural-product-knowledge-table`** | Prior / Knowledge | [Kaggle](https://www.kaggle.com/datasets/dmitriigluzdov/casmi26-natural-product-knowledge-table) | `natural_products.parquet`, `lotus_structure_genus.parquet` | 152k natural products from LOTUS + NPAtlas with taxonomic trees and superclasses. |
| **`ahmedberatozer/casmi26-iceberg`** | MS Simulator | [Kaggle](https://www.kaggle.com/datasets/ahmedberatozer/casmi26-iceberg) | `ice_runner.py`, `ice_fuse.py`, checkpoint | Physics-based forward mass-spec simulator for formula isomer re-ranking. |
| **`ahmedberatozer/casmi26-glacier`** | MS Simulator | [Kaggle](https://www.kaggle.com/datasets/ahmedberatozer/casmi26-glacier) | `gl_runner.py`, `gl_fuse.py`, checkpoint | Transformer-based isomer re-scorer, restricted to `[M+H]+` spectra (+0.006 to +0.008 LB). |
| **`megayak/casmi26-simulated-ranker-rows`** | Training Data | [Kaggle](https://www.kaggle.com/datasets/megayak/casmi26-simulated-ranker-rows) | `sim_rank_rows_nofp.npz`, `sim_rank_rows_fp16k.npz` | 399k leak-free training rows for GBDT candidate ranker training. |
| **`hengck23/hengck23-dreams-enveda-casmi26`** | Candidate Pool / Tail | [Kaggle](https://www.kaggle.com/datasets/hengck23/hengck23-dreams-enveda-casmi26) | `dreams_np_mass.npy`, `dreams_np_smiles.npy` | Precomputed DreaMS NP mass & SMILES arrays for mass-window candidate tail backfill up to rank 25. |
| **`prvsiyan/chebi-lipidmaps-casmi26`** | Candidate Pool | [Kaggle](https://www.kaggle.com/datasets/prvsiyan/chebi-lipidmaps-casmi26) | `bio_fp.npy`, `bio_mass.npy`, `bio_meta.pkl` | ChEBI + LIPID MAPS candidates for Engine 2 candidate expansion (BIO channel). |
| **`prvsiyan/coconut-casmi26-candidates`** | Candidate Pool | [Kaggle](https://www.kaggle.com/datasets/prvsiyan/coconut-casmi26-candidates) | `coco_fp.npy`, `coco_mass.npy`, `coco_meta.pkl`, `fp_bits.npy` | COCONUT candidate pool with 6,930-bit packed fingerprints; provides `fp_bits.npy` for Engine 2. |
| **`prvsiyan/casmi26-ranker-features`** | Training Data | [Kaggle](https://www.kaggle.com/datasets/prvsiyan/casmi26-ranker-features) | `rank_train.npz` | Shipped ranker training rows for Engine 2 ranker blending. |
