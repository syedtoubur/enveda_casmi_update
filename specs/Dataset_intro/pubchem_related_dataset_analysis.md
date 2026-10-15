# CASMI 2026: Comprehensive PubChem & Natural Product Dataset Master Guide

> **Target File**: [`specs/Dataset_intro/pubchem_related_dataset_analysis.md`](file:///e:/ENVEDA_casmi/specs/Dataset_intro/pubchem_related_dataset_analysis.md)  
> **Domain**: Enveda CASMI 2026 — Molecule ID from LC-MS/MS Mass Spectra  
> **Evaluation Metric**: Mean Reciprocal Rank at 25 (`MRR@25`) evaluated on tautomer-canonical `InChIKey14`  
> **Objective**: Provide an authoritative, self-contained reference on all PubChem-related candidate datasets, natural product knowledge tables, cheminformatics filters, and popularity priors used across the state-of-the-art competitive lineage.

---

## Table of Contents
1. [Executive Philosophy: Candidate Coverage vs. Decoy Dilution](#1-executive-philosophy)
2. [Master Dataset Inventory & Technical Specifications](#2-master-dataset-inventory)
3. [Dataset 1: `casmi26-pubchem-tier` (105.88M Structures)](#3-dataset-1-casmi26-pubchem-tier)
   - [Origin & Filtering Criteria](#31-origin--filtering-criteria)
   - [The 3-File Memory-Mapped Architecture](#32-the-3-file-memory-mapped-architecture)
   - [The Two-Pass Retrieval Cascade (`probe_core2.py`)](#33-the-two-pass-retrieval-cascade)
   - [Confidence Gated Interleaving](#34-confidence-gated-interleaving)
   - [Code Recipe: Building a Memory-Mapped Tier from Scratch](#35-code-recipe-building-a-tier)
4. [Dataset 2: `casmi26-pubchem-popularity-prior` (Version 2)](#4-dataset-2-casmi26-pubchem-popularity-prior)
   - [Origins & Construction](#41-origins--construction)
   - [Schema & Row Alignment](#42-schema--row-alignment)
   - [The Popularity Formulation & Leaderboard Lift](#43-the-popularity-formulation)
5. [Dataset 3: PubChemLite & The 6,930-Bit Precomputed Artifact](#5-dataset-3-pubchemlite--6930-bit-artifact)
   - [What PubChemLite Is (Schymanski et al.)](#51-what-pubchemlite-is)
   - [The 385,760 Candidate Artifact (`cosmicbe/casmi26-pubchemlite`)](#52-the-385760-candidate-artifact)
   - [Dual-Engine Fingerprint Compatibility: 10,226 vs. 6,930 Bits](#53-dual-engine-fingerprint-compatibility)
   - [Integration Strategy into Engine 2](#54-integration-strategy-into-engine-2)
6. [Dataset 4: `casmi26-natural-product-knowledge-table` (`natural_products.parquet`)](#6-dataset-4-natural-product-knowledge-table)
   - [Source Composition: LOTUS + NPAtlas](#61-source-composition)
   - [The 2,715 Missing Pool Structures](#62-the-2715-missing-pool-structures)
   - [Plant Taxonomy & Biosynthetic Pathways](#63-plant-taxonomy--biosynthetic-pathways)
7. [Cheminformatics Curation & Natural Product Discrimination](#7-cheminformatics-curation)
   - [Natural vs. Synthetic Chemical Space](#71-natural-vs-synthetic-space)
   - [Physicochemical Signatures ($Fsp^3$, Heteroatoms, Ertl NP-Score)](#72-physicochemical-signatures)
   - [The Two-Tier Stratified Architecture (Zero Recall Loss)](#73-stratified-architecture)
8. [Comprehensive Python Recipe Suite](#8-comprehensive-python-recipe-suite)

---

## 1. Executive Philosophy

In the CASMI 2026 competition, 100% of test spectra originate from **biological extracts (primarily medicinal plants and microbial cultures)**. In silico small-molecule identification operates under three distinct novelty classes:

```
                                CASMI 2026 TEST COHORT
                                          │
        ┌─────────────────────────────────┼─────────────────────────────────┐
        ▼                                 ▼                                 ▼
   Class 1: Library Match            Class 2: Database Match           Class 3: Completely Novel
   • Reference spectrum in train     • Structure in PubChem/COCONUT    • No database match exists
   • Cosine similarity >= 0.90       • No reference spectrum           • Needs de novo / mutator
   • Pure spectral match wins        • **Retrieval ceiling problem**    • Retrieval alone = 0%
```

### The Candidate Coverage Dilemma
* **The Ceiling Problem**: The default competition pool (`casmi26-v2-pool`) indexes **710,701 structures** (Train + COCONUT 2.0). If a test molecule is an obscure plant secondary metabolite documented in PubChem but omitted from COCONUT, any retrieval model restricted to the 710k pool scores **MRR@25 = 0.000**.
* **The Dilution Problem**: PubChem contains **>115 million entries**. However, over **98%** are combinatorial screening libraries (Enamine REAL, ChemBridge), virtual compounds, petroleum derivatives, and industrial dyes. Slicing a $\pm 10$ ppm mass window in raw PubChem yields 10,000–50,000 isomers, diluting the true natural product.
* **The Solution**: Maintain a **100% recall ceiling** via ultra-fast, zero-RAM memory-mapped PubChem search, while applying **continuous biological and cheminformatics priors** so true natural products float to the top 25 candidate slots.

---

## 2. Master Dataset Inventory

| Dataset Identifier | Author / Source | Rows / Entities | Disk Size | Core Files & Types | Fingerprint Basis | Primary Role in Pipeline |
|---|---|---|---|---|---|---|
| **`casmi26-pubchem-tier`** | Ahmed Berat Özer | 105,875,489 structures | ~7.2 GB | `pc_mass.npy` (f64)<br>`pc_off.npy` (i64)<br>`pc_smiles.npy` (u8) | Standard ECFP4 (4,096) + Full (10,226) on the fly | High-speed, zero-RAM candidate retrieval for Class 2. |
| **`casmi26-pubchem-popularity-prior`** (v2) | Dmitrii Gluzdov | 105,875,489 rows (aligned 1:1) | ~1.5 GB | `pc_lsid.npy` (f16)<br>`pc_lpmid.npy` (f16)<br>`pc_lpatent.npy` (f16)<br>`pc_cid.npy` (i32)<br>`pc_ik14.npy` (S14)<br>`pc_np.npy` (u8) | N/A (Metadata / Priors) | Ranks candidates by PubMed, substance deposits, and NP flags (+0.012 LB). |
| **PubChemLite 6,930-bit Artifact** | `cosmicbe` (derived from Schymanski et al.) | 385,760 structures | ~330 MB | Precomputed 6,930-bit packed fingerprints + exact masses | **6,930 bits** (prvsiyan basis) | Direct expansion candidate pool for **Engine 2**. |
| **`casmi26-natural-product-knowledge-table`** | Community (LOTUS + NPAtlas) | 152,717 connectivities | ~25 MB | `natural_products.parquet`<br>`lotus_structure_genus.parquet` | RDKit 2026.03.3 `scorer_key` | Provides plant taxonomy, NPClassifier superclasses, and 2,715 missing pool hits. |
| **`casmi26-v2-pool`** | Ahmed Berat Özer | 710,701 structures | ~1.2 GB | `pool_meta.parquet`<br>`pool_fp.npy`<br>`pool_frag_*.npy` | **10,226 bits** (`fp_bits.npy`) | Primary candidate pool for **Engine 1 (v4n)**. |

---

## 3. Dataset 1: `casmi26-pubchem-tier` (105.88M Structures)

### 3.1 Origin & Filtering Criteria
Derived from the official NCBI PubChem `Compound/Extras` dump (`CID-SMILES.gz` and `CID-Mass.gz`).
1. **Mass Range**: 150.0 Da to 1,250.0 Da neutral monoisotopic mass. Covers >99.9% of small-molecule secondary metabolites and LC-MS detectable molecules.
2. **Elemental Composition**: Organic CHNOPS + Halogens (F, Cl, Br, I). All transition metals, organometallics, and inorganic salts stripped.
3. **2D Graph Canonicalization**: All stereochemical flags (`@`, `@@`, `/`, `\`) removed. Collapsed into flat 2D connectivity blocks because the competition metric evaluates tautomer-canonical InChIKey14.
4. **Monotonic Sorting**: Sorted strictly ascending by exact neutral monoisotopic mass.

### 3.2 The 3-File Memory-Mapped Architecture
To query 105.88 million SMILES strings inside Kaggle's 16 GB RAM and 9-hour limit, the author developed a binary memory-mapped (`np.memmap`) layout:

```
[pc_mass.npy]  (float64, 847 MB)  ──► Sorted array of exact neutral masses.
                                      Enables O(log N) binary search via np.searchsorted.

[pc_off.npy]   (int64,   847 MB)  ──► Byte pointers into pc_smiles.npy.
                                      Molecule i begins at off[i] and ends at off[i+1].

[pc_smiles.npy](uint8,  5.52 GB)  ──► Flat continuous ASCII byte array.
                                      Loaded via mmap_mode='r'. Consumes 0 MB RAM until sliced.
```

When querying a $\pm 10$ ppm mass window, the operating system page cache loads **only the exact disk sectors containing the matching SMILES** (a few megabytes).

### 3.3 The Two-Pass Retrieval Cascade (`probe_core2.py`)
Evaluating 10,226-bit fingerprints on 50,000 retrieved PubChem SMILES on CPU would breach the 9-hour time limit. The pipeline uses a **2-Pass Cascade**:

```
                         Precursor Mass Target (± 10 ppm)
                                        │
                                        ▼
                   np.searchsorted on pc_mass.npy (20 µs)
                   Slices ~5,000 to 40,000 raw SMILES
                                        │
                                        ▼
             [Pass 1: Rapid Coarse Screen (Standard 4,096-bit ECFP4)]
             • Uses RDKit C++ MorganGenerator(radius=2, fpSize=4096)
             • Dot product: sc1 = E @ z_ecfp4
             • np.argpartition retains top PC_N1 = 5,000 candidates
                                        │
                                        ▼
             [Pass 2: Bayes Fine Scoring (Full 10,226-bit Fingerprints)]
             • Computes full topological, path, and MACCS bits
             • Exact log-likelihood dot product: fz = f @ z_full
             • Sort descending
                                        │
                                        ▼
             [Deduplication & Main Pool Exclusion]
             • Computes tautomer-canonical InChIKey14
             • Discards any structure already in the 710k Main Pool
             • Discards duplicate tautomers/stereoisomers
             • Retains top 25 PubChem-only candidates
```

### 3.4 Confidence Gated Interleaving
PubChem candidates are merged into the final submission via a dynamic confidence gate:
* **Library Protection (`lib_max >= 0.90`)**: If the query has a high-confidence spectral cosine match against `train.parquet`, the molecule is a Class 1 library hit. **PubChem candidates are completely suppressed**.
* **Aggressive Mode ($\Delta_{\text{rel}} > 600$)**: If PubChem's top candidate significantly outperforms the main pool candidate ($\Delta_{\text{rel}} = \text{top\_PubChem\_fz} - \text{top\_Pool\_fz} > 600$), it is inserted into high-priority slots:
  $$\text{Slots} = [2, 4, 6, 8, 10]$$
* **Gentle Mode ($\Delta_{\text{rel}} \le 600$)**: If the score difference is modest, PubChem candidates are interleaved conservatively:
  $$\text{Slots} = [4, 8, 12, 16, 20]$$

---

## 4. Dataset 2: `casmi26-pubchem-popularity-prior` (Version 2)

### 4.1 Origins & Construction
Authored by Dmitrii Gluzdov, this dataset extracts official documentation counts from the NCBI PubChem release of 2026-09-27 (`CID-SID.gz`, `CID-PMID.gz`, `CID-Patent.gz`, `CID-SMILES.gz`, `CID-InChI-Key.gz`) and maps them **row-by-row** to `casmi26-pubchem-tier` (105,875,489 rows) and `casmi26-v2-pool` (710,701 rows).

### 4.2 Schema & Row Alignment

| File Name | Rows | Data Type | Physical Meaning |
|---|---|---|---|
| `pc_lsid.npy` | 105,875,489 | `float16` | $\log(1 + \text{PubChem Substance records deposited})$ |
| `pc_lpmid.npy` | 105,875,489 | `float16` | $\log(1 + \text{Distinct PubMed research articles})$ |
| `pc_lpatent.npy`| 105,875,489 | `float16` | $\log(1 + \text{Patent documents mentioning structure})$ |
| `pc_cid.npy` | 105,875,489 | `int32` | Representative PubChem CID (the most-deposited stereoisomer) |
| `pc_ik14.npy` | 105,875,489 | `string (S14)`| InChIKey connectivity block of that representative CID |
| `pc_np.npy` | 105,875,489 | `uint8` | Bitmask: $1 = \text{COCONUT}$, $2 = \text{LOTUS}$, $4 = \text{NPAtlas}$ |
| `pool_*.npy` | 710,701 | Same types | Identical annotations aligned to the 710k candidate pool |

### 4.3 The Popularity Formulation & Leaderboard Lift
A compound that appears in an LC-MS extract has usually been isolated, synthesized, or studied. In contrast, virtual combinatorial screening decoys have $\text{SID} = 1$, $\text{PMID} = 0$, and $\text{Patent} = 0$.

In Version 4 of the competitive lineage (`v29` / `0.411 LB`), candidates are re-ordered by combining spectral likelihood with the log-popularity prior:
$$\text{Score}_{\text{final}} = z(\text{Score}_{\text{model}}) + 0.25 \cdot \text{pc\_lsid} + 0.15 \cdot \text{pc\_lpmid} + 1.0 \cdot (\text{pc\_np} > 0)$$

* **Score Impact**: Adding this prior directly inside the PubChem channel (`V17_PATCH`, $\lambda=0.25$, `POP_UNION=200`) delivered a **+0.006 to +0.010 LB increase**, pushing the single-engine ceiling from 0.399 to **0.411+**.

---

## 5. Dataset 3: PubChemLite & The 6,930-Bit Precomputed Artifact

### 5.1 What PubChemLite Is
Published by Schymanski et al. (*J. Cheminf.* 2021), PubChemLite is an expert-curated subset of PubChem (~460,000 parent structures) filtered by 10 biological and environmental categories (BioPathway, DrugMedicInfo, FoodRelated, DisorderDisease, etc.). In benchmark trials, incorporating PubChemLite annotation counts into MetFrag improved Top-1 identification accuracy from **58% to 70.5%**.

### 5.2 The 385,760 Candidate Artifact (`cosmicbe/casmi26-pubchemlite`)
The Kaggle artifact consists of PubChemLite compounds **absent from COCONUT 2.0**:
$$\text{PubChemLite (460k)} \setminus \text{COCONUT (436k)} = \mathbf{385,760 \text{ unique parent structures}}$$
* **Contents**: Exact neutral monoisotopic masses and precomputed **6,930-bit RDKit binary fingerprints** generated with RDKit 2026.03.3.

### 5.3 Dual-Engine Fingerprint Compatibility: 10,226 vs. 6,930 Bits
Your competitive pipeline utilizes two distinct retrieval engines:

```
                            THE TWO RETRIEVAL ENGINES
                                        │
        ┌───────────────────────────────┴───────────────────────────────┐
        ▼                                                               ▼
  Engine 1: v4n (10,226-bit basis)                                Engine 2: prvsiyan (6,930-bit basis)
  • Basis: fp_bits.npy (10,226 bits)                              • Basis: pv_fp.py (6,930 bits)
  • Neural Net: fpnet_full1.pt (10,226 logits)                    • Neural Net: casmi26-fp-models-v2 (6,930 logits)
  • Incompatible with 6,930-bit precomputations                   • NATIVE 1:1 DROP-IN MATCH
```

### 5.4 Integration Strategy into Engine 2
Because the 6,930 bits match Engine 2's neural model natively, you can expand Engine 2's pool in `casmi_engine.py`:
```python
# Load COCONUT + BIO
co_fp = np.load("coco_fp.npy") # 6930 bits
bio_fp = np.load("bio_fp.npy") # 6930 bits

# Load PubChemLite non-COCONUT artifact
pcl_fp = np.load("pubchemlite_6930_fp.npy") # 385,760 x 867 bytes
pcl_mass = np.load("pubchemlite_mass.npy")

# Unified pool: 710k -> 1.18M structures
all_fp = np.vstack([co_fp, bio_fp, pcl_fp])
all_mass = np.concatenate([co_mass, bio_mass, pcl_mass])
```
* **Advantage**: These 385,760 structures are now evaluated by Engine 2's **31-feature GBDT ranker**, receive **AFIX spectral analog propagation**, and enter the **ICE_UNION for forward physics simulation (ICEBERG/GLACIER)**.

---

## 6. Dataset 4: `casmi26-natural-product-knowledge-table` (`natural_products.parquet`)

### 6.1 Source Composition
Compiled from **LOTUS (frozen metadata 2023-01-06)** and **NPAtlas (2026-09-30)**, standardized with RDKit 2026.03.3:
* **Total Connectivities**: 152,717 unique natural products.
* **Class-1 Coverage**: Only **8.2% (12,547 structures)** exist in the training library.
* **Class-2 Pool Coverage**: 98% exist in the 710k candidate pool.
* **Missing Pool Structures**: Exactly **2,715 natural products are missing from the 710k public pool**!

### 6.2 The 2,715 Missing Pool Structures
In Track 4 probe P3e ([probe3_plan.md](file:///e:/ENVEDA_casmi/specs/track4_priors_probes/probe3_plan.md)), these 2,715 structures are dynamically retrieved:
```python
npt = pd.read_parquet("natural_products.parquet")
missing_nps = npt[(~npt.in_casmi_pool.astype(bool)) & (npt.mono_mass.between(150.0, 1250.0))]
```
Injecting these into Engine 2 eliminates the 0% recall ceiling on unindexed LOTUS natural products.

### 6.3 Plant Taxonomy & Biosynthetic Pathways
* **Taxonomic Distribution**: ~94,000 structures are produced exclusively by **Archaeplastida / Viridiplantae (green plants)**.
* **Pathway Annotations**: Contains NPClassifier classifications across 7 pathways (Terpenoids, Shikimates, Alkaloids, Polyketides, etc.).
* **Ubiquity Metric**: `n_organisms` indicates how widely distributed the compound is across species.

---

## 7. Cheminformatics Curation & Natural Product Discrimination

### 7.1 Natural vs. Synthetic Chemical Space
Combinatorial screening decks are assembled using automated parallel coupling reactions (amide formations, Suzuki couplings). Natural products are synthesized via enzymatic cascade pathways.

```
       PROPERTY                    SYNTHETIC COMBINATORIAL              NATURAL PRODUCTS (PLANT/BIO)
Fraction sp3 (Fsp3)                Flat, aromatic (< 0.25)              Stereo-rich, tetrahedral (>= 0.40)
Heteroatom Economy                 Nitrogen-rich (Amides, Triazoles)    Oxygen-rich (Polyols, Lactones, Esters)
Halogen Content                    Frequently Cl, Br, F                 Halogens rare (except marine)
Ring Topology                      Simple monocyclic/bicyclic           Fused, bridged, spiro, macrocyclic
Ertl NP-Likeness Score             -2.5 to -0.5                         +0.5 to +4.0
```

### 7.2 Physicochemical Signatures & Rules
1. **The Polyphenol / Flavonoid Rule**: Flavonoids have low $Fsp^3$ due to their aromatic rings, but have high oxygenation:
   $$\text{If } Fsp^3 < 0.25 \implies \frac{N_{\text{Oxygen}}}{N_{\text{Carbon}}} \ge 0.28 \quad \text{and} \quad N_{\text{Nitrogen}} = 0$$
2. **The Synthetic Amide Trap**: A candidate is almost certainly a synthetic decoy if:
   $$\frac{N_{\text{Nitrogen}}}{N_{\text{Carbon}}} > 0.20 \quad \text{and} \quad \frac{N_{\text{Oxygen}}}{N_{\text{Carbon}}} < 0.15$$
3. **Ertl Natural Product-Likeness Score**:
   $$f_i = \log\left(\frac{n_{\text{NP}, i}}{n_{\text{Synth}, i}} \times \frac{N_{\text{Synth}, \text{total}}}{N_{\text{NP}, \text{total}}}\right), \quad \text{Score} = \frac{1}{|A|} \sum_i f_i$$

### 7.3 Stratified Two-Tier Architecture (Zero Recall Loss)
Do **not** delete synthetic compounds from your master PubChem tier. Instead, use a two-tier strategy:
* **Tier 1 (Curated Bio-Core, ~2M compounds)**: Evaluated first with priority candidate slots. Formed by compounds with `pc_np > 0`, `pc_lpmid > 0`, or $\text{NP\_Score} \ge 0.50$.
* **Tier 2 (Full 105.88M PubChem Universe)**: Acts as a fallback for novel or uncataloged chemistry.

---

## 8. Comprehensive Python Recipe Suite

### Recipe 1: Building a Memory-Mapped Tier from Raw SMILES
```python
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt

def serialize_custom_tier(smiles_iterable, out_prefix="custom_tier"):
    """
    Filters, strips stereochemistry, mass-sorts, and serializes
    molecules into the 3-file memory-mapped architecture.
    """
    records = []
    allowed_atoms = {1, 6, 7, 8, 15, 16, 9, 17, 35, 53} # CHNOPS + Halogens

    for smi in smiles_iterable:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        if not all(a.GetAtomicNum() in allowed_atoms for a in m.GetAtoms()):
            continue

        # Strip stereochemistry for InChIKey14 evaluation
        Chem.RemoveStereochemistry(m)
        clean_smi = Chem.MolToSmiles(m, isomericSmiles=False)
        mw = float(CalcExactMolWt(m))

        if 150.0 <= mw <= 1250.0:
            records.append((mw, clean_smi))

    # Sort strictly by monoisotopic mass
    records.sort(key=lambda x: x[0])
    masses = np.array([r[0] for r in records], dtype=np.float64)
    smis = [r[1].encode("ascii") for r in records]

    # Calculate byte offsets
    lengths = [len(s) for s in smis]
    offsets = np.zeros(len(smis) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)

    # Pack flat byte buffer
    total_bytes = offsets[-1]
    raw_buf = np.empty(total_bytes, dtype=np.uint8)
    curr = 0
    for s_bytes in smis:
        raw_buf[curr:curr + len(s_bytes)] = np.frombuffer(s_bytes, dtype=np.uint8)
        curr += len(s_bytes)

    # Save to disk
    np.save(f"{out_prefix}_mass.npy", masses)
    np.save(f"{out_prefix}_off.npy", offsets)
    np.save(f"{out_prefix}_smiles.npy", raw_buf)
    print(f"Serialized {len(masses):,} structures into {out_prefix}. Raw buffer: {total_bytes / (1024**2):.2f} MB")
```

---

### Recipe 2: High-Speed Slicing of Memory-Mapped Candidates
```python
import numpy as np

def slice_mass_window(target_mass, ppm_tol, pc_dir):
    """
    Extracts SMILES strings within a mass window in microseconds.
    """
    mass = np.load(f"{pc_dir}/pc_mass.npy", mmap_mode="r")
    off = np.load(f"{pc_dir}/pc_off.npy", mmap_mode="r")
    buf = np.load(f"{pc_dir}/pc_smiles.npy", mmap_mode="r")

    tol = target_mass * ppm_tol * 1e-6
    a = int(np.searchsorted(mass, target_mass - tol, "left"))
    b = int(np.searchsorted(mass, target_mass + tol, "right"))
    if b <= a:
        return []

    sub_off = np.asarray(off[a:b + 1], dtype=np.int64)
    raw = bytes(buf[sub_off[0]:sub_off[-1]])
    base = sub_off[0]

    return [raw[sub_off[i] - base:sub_off[i + 1] - base].decode("ascii") for i in range(b - a)]
```

---

### Recipe 3: Batch NP-Likeness Scoring with RDKit
```python
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Contrib.NP_Score import npscorer

fscore = npscorer.readNPModel()

def calculate_np_metrics(smi: str):
    """
    Computes Fsp3, heteroatom ratios, and Ertl NP-Score.
    """
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return dict(valid=False, np_score=-5.0, fsp3=0.0, o_c=0.0, n_c=0.0)

    c_cnt = sum(1 for a in m.GetAtoms() if a.GetAtomicNum() == 6)
    o_cnt = sum(1 for a in m.GetAtoms() if a.GetAtomicNum() == 8)
    n_cnt = sum(1 for a in m.GetAtoms() if a.GetAtomicNum() == 7)

    fsp3 = Descriptors.FractionCSP3(m)
    score = npscorer.scoreMol(m, fscore)

    return dict(
        valid=True,
        np_score=score,
        fsp3=fsp3,
        o_c=o_cnt / max(c_cnt, 1),
        n_c=n_cnt / max(c_cnt, 1),
        is_flavonoid_like=(fsp3 < 0.25 and (o_cnt / max(c_cnt, 1)) >= 0.28 and n_cnt == 0),
        is_terpenoid_like=(fsp3 >= 0.45 and n_cnt <= 1)
    )
```

---

### Recipe 4: Constructing a Plant Taxonomic Prior Lookup
```python
import numpy as np
import pandas as pd

def build_plant_prior_map(parquet_path="natural_products.parquet"):
    """
    Builds an O(1) in-memory dictionary mapping InChIKey14 -> plant prior bonus.
    """
    df = pd.read_parquet(parquet_path, columns=["scorer_key", "kingdoms", "n_organisms", "pubmed_articles"])

    is_plant = df["kingdoms"].fillna("").str.contains("Archaeplastida|Viridiplantae")
    
    # Continuous weighting: plant bonus + ubiquity + documentation
    prior_score = (
        1.0 * is_plant.astype(float) +
        0.25 * np.log1p(df["n_organisms"].fillna(0)) +
        0.15 * np.log1p(df["pubmed_articles"].fillna(0))
    )

    prior_map = dict(zip(df["scorer_key"], prior_score))
    print(f"Constructed plant prior dictionary with {len(prior_map):,} entries.")
    return prior_map
```
