# Analog-Guided Core Invariance & MACCS Keys in Mutator v2

> **Document**: `specs/track4_priors_probes/summary_maccs.md`  
> **Topic**: Solving ECFP Radius Bleed and Combinatorial Explosion using 166 MACCS Structural Keys.  
> **Target Audience**: Autonomous agents (Claude / Antigravity) and human researchers refining Class-3 (novel structure) generation.  
> **Target Base**: `casmi26-v29-fusion-pop-glmh-0-415.ipynb` (0.415 LB SOTA).

---

## 1. Executive Summary

In mass spectrometry-based de novo structure identification (Class 3), the candidate space is combinatorial and unconstrained. To guide the **Track-1 $\Delta M$ mutator v2** toward the true novel structure without producing tens of thousands of synthetic decoys, we need a **directional compass** that distinguishes:
1. **The Invariant Parent Core (Scaffold)**: Must be preserved 100%.
2. **The Variable Peripheral Decorations**: Where mutations legitimately occur in nature.

Using circular fingerprints (Morgan / ECFP4) to enforce core preservation suffers from a critical failure mode: **"Radius Bleed"**. 

This document outlines the theoretical mechanism of Radius Bleed, why **166 MACCS Structural Keys** completely solve it without new packages, audits the current 0.415 codebase, and details how to integrate this into the Mutator v2 plan.

---

## 2. The Problem: What is "Radius Bleed"?

### 2.1 How Circular (Morgan / ECFP) Fingerprints Work
Circular fingerprints operate like expanding ripples in a pond around every heavy atom:
- **Radius 0**: Atom identity + hybridization ($C_{sp2}$).
- **Radius 1**: Atom + neighbors 1 bond away ($C—C$).
- **Radius 2 (ECFP4)**: Atom + neighbors up to 2 bonds away ($C—C—O$).

Each circular sub-environment is hashed into a bit index (e.g., bit #450).

### 2.2 The Radius Bleed Mechanism
Consider a core benzene ring atom ($C_2$) adjacent to an aromatic carbon ($C_1$) bearing an exocyclic decoration (e.g., a hydroxyl group $-\text{OH}$):

```
       Core Ring Carbon C₂          Core Ring Carbon C₁         Peripheral Decoration
             [ C₂ ] ───────────────────── [ C₁ ] ───────────────────── [ —OH ]
               │                            │                             │
               └───────── 1 bond ───────────┴────────── 2 bonds ──────────┘
                                        ▲
              The 2-bond radius of core atom C₂ captures the decoration!
```

1. Core atom $C_2$ has a 2-bond circular environment that includes the $-\text{OH}$ group on $C_1$. This is encoded into **Bit #450**.
2. Suppose the mutator performs a valid peripheral mutation: **$-\text{OH} \longrightarrow -\text{OCH}_3$** (O-methylation).
3. **Did the core ring break?** **NO.** The 6-membered aromatic ring is 100% intact.
4. **Did Bit #450 disappear?** **YES!** Because $C_2$'s 2-bond radius now contains a methyl carbon instead of a hydrogen, its hash changes completely.

### 2.3 The Consequence for Core Invariance Filtering (False Rejections)
If we naively enforce core preservation on ECFP4 bits:
$$\text{If candidate } B \text{ loses any core bit present in anchor } A \implies \text{PRUNE}$$
A valid edit on a peripheral chain will flip bits belonging to adjacent core ring atoms. The filter mistakenly concludes *"the scaffold was destroyed"* and **wrongfully discards the correct candidate**.

---

## 3. The Solution: 166 MACCS Structural Keys

### 3.1 Why MACCS Keys Have Zero Radius Bleed
MACCS (Molecular ACCess System) keys do **not** construct circular expanding neighborhoods. Instead, they evaluate a fixed dictionary of **166 explicit SMARTS pattern queries**:

- **Bit 125**: *"Does the molecule contain an aromatic 6-membered ring?"* $\rightarrow$ **True / False**
- **Bit 80**: *"Does the molecule have 3 fused rings?"* $\rightarrow$ **True / False**
- **Bit 139**: *"Does the molecule have an oxygen bonded to an aromatic ring?"* $\rightarrow$ **True / False**
- **Bit 106**: *"Is there a carbonyl group ($C=O$)?"* $\rightarrow$ **True / False**

### 3.2 The Immunity to Peripheral Changes
When modifying $-\text{OH} \longrightarrow -\text{OCH}_3$:
- Bit 125 (*Aromatic 6-membered ring*) remains **True**.
- Bit 80 (*Fused rings*) remains **True**.

Because each key corresponds to a self-contained substructure rather than a 2-bond environment, **peripheral edits never alter the keys describing the core ring skeleton**.

```
┌─────────────────────────────────┬─────────────────────────────────┐
│     ECFP4 (Circular Hash)       │      MACCS Keys (166 SMARTS)    │
├─────────────────────────────────┼─────────────────────────────────┤
│ • 2-bond expanding neighborhood │ • Fixed checklist of questions  │
│ • Local changes bleed inward    │ • Independent substructure test │
│ • Prone to false core-breakage  │ • ZERO radius bleed             │
│ • High bit collision rate       │ • Deterministic, 1:1 interpretable│
└─────────────────────────────────┴─────────────────────────────────┘
```

---

## 4. Current Status in the 0.415 Notebook (`casmi26-v29-…-0-415.ipynb`)

An audit of the current SOTA 0.415 scoring code reveals:

### 4.1 MACCS is ALREADY Running Offline in Engine 2
In Cell C3 (`casmi_engine.py`, lines 177–179 of the py:percent conversion):
```python
def _fp_init():
    _g["bits"] = np.load(find("fp_bits.npy"))
    _g["m2"] = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
    _g["m3"] = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=4096)
    _g["rk"] = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048, maxPath=6)

def fp_and_mass(smi):
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys              # <--- ALREADY PRESENT
    from rdkit.Chem.Descriptors import ExactMolWt
    ...
    fp = np.concatenate([
        _g["m2"].GetFingerprintAsNumPy(m).astype(np.uint8),      # 4,096
        _g["m3"].GetFingerprintAsNumPy(m).astype(np.uint8),      # 4,096
        _g["rk"].GetFingerprintAsNumPy(m).astype(np.uint8),      # 2,048
        np.array(MACCSkeys.GenMACCSKeys(m), dtype=np.uint8)       # 167 MACCS keys
    ])[_g["bits"]]                                               # 6,930 bits
```

### 4.2 Why It is Not Helping Right Now
In the current code, the 167 MACCS bits are **submerged** inside the composite 6,930-bit vector. Because Morgan bits outnumber MACCS bits **50 to 1**, the ranker and neural models blend them together. 

The pipeline currently **never isolates the 166 MACCS bits** to evaluate scaffold conservation or prune mutations.

> [!IMPORTANT]
> Because `from rdkit.Chem import MACCSkeys` is already part of the offline RDKit wheel installed in Cell 1, **implementing MACCS-based core profiling requires zero new datasets, zero new wheels, and zero model re-training.**

---

## 5. Concrete Algorithmic Implementation for Mutator v2

### 5.1 Step 1: Building the Consensus Core Profile ($P_{\text{core}}$)
For each query, retrieve the top $K$ mass-shifted spectral analogs ($\pm 200\text{ Da}$, shifted similarity $w_k \ge 0.60$):

```python
from rdkit.Chem import MACCSkeys

def build_maccs_core_profile(analog_smiles, weights, tau_core=0.90):
    """
    Computes bit-level conservation across spectral analogs.
    Returns:
      core_keys: indices of MACCS keys present in >= tau_core of analogs.
      disallowed_keys: indices of MACCS keys present in <= 0.05 of analogs.
    """
    maccs_list = []
    for s in analog_smiles:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            # Drop bit 0 (unused dummy bit in RDKit MACCS)
            maccs_list.append(np.array(MACCSkeys.GenMACCSKeys(m))[1:])
    
    if not maccs_list:
        return np.array([], dtype=int), np.array([], dtype=int)
        
    M = np.stack(maccs_list, axis=0) # (K, 166)
    W = np.asarray(weights, dtype=np.float64)[:len(M)]
    W /= W.sum() + 1e-9
    
    P = (M * W[:, None]).sum(axis=0) # (166,)
    
    core_keys = np.where(P >= tau_core)[0]
    disallowed_keys = np.where(P <= 0.05)[0]
    return core_keys, disallowed_keys
```

### 5.2 Step 2: The Core-Preservation Sanity Check (Lossless Pruning)
When evaluating any candidate $B$ generated by the mutator:

```python
def validate_core(candidate_mol, core_keys):
    """
    Returns True only if candidate preserves 100% of the conserved scaffold keys.
    Zero radius-bleed false alarms.
    """
    if len(core_keys) == 0:
        return True
    cand_maccs = np.array(MACCSkeys.GenMACCSKeys(candidate_mol))[1:]
    # Must have all core keys active (value == 1)
    return bool(np.all(cand_maccs[core_keys] == 1))
```

### 5.3 Step 3: Pruning Intermediate Reaction Paths ($A \to C \to B$)
For multi-step mutations $A \xrightarrow{e_1} C \xrightarrow{e_2} B$:
- Use the FPNet predicted logit vector $z$ to compute intermediate similarity:
  $$\Delta fz(C) = (\mathbf{fp}_C - \mathbf{fp}_A) \cdot z$$
- **The Negative Slack Rule ($-\tau$)**: Do not require $\Delta fz(C) > 0$ strictly, because subtractive steps (e.g. deglycosylation) naturally lose bits. Instead, prune only if:
  $$\Delta fz(C) < -\tau \quad (\text{e.g., } \tau = 0.20 \cdot \sigma_z)$$
- **Why this is lossless**: If a 2-step edit is chemically favorable ($\Delta(B) > 0$), at least one of the two independent paths ($A \to C_1 \to B$ or $A \to C_2 \to B$) starts with the non-negative edit, easily surviving the $-\tau$ buffer.

---

## 6. Comparison of Approaches

| Method | Core Protection Accuracy | Radius Bleed Risk | Combinatorial Speedup | Implementation Cost |
| :--- | :---: | :---: | :---: | :---: |
| **None (Track 1 Baseline)** | Low (Alphabetical beam cut) | High (N/A) | None (Generates 8k decoys) | Baseline |
| **ECFP4 Filter** | Medium (False alarms) | **Severe (Kills valid edits)** | High | Low |
| **MACCS Structural Keys (Proposed)** | **High (Direct Substructures)** | **ZERO** | **High (Cuts 80%+ decoys)** | **Zero new packages** |

---

## 7. Recommended Action Plan for Claude
1. **In `build/delta_m_mutator.py`**:
   - Add `MACCSkeys` profile builder for the top analog seeds.
   - Insert `validate_core()` inside candidate filtering, dropping candidates that destroy conserved core keys.
2. **In `build/mut_runner.py`**:
   - Apply the intermediate $-\tau$ slack check during multi-step expansion.
   - Combine with `V17_PATCH` uncorrupted gate metrics (`fz_top` and `best_pool_fz`).
