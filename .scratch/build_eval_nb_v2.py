import json
from pathlib import Path

cells = []

def add_md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

def add_code(text):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

add_md("""# CASMI 2026: Decoy Autopsy & Smart Filter Gate Evaluation
### Eliminating False Positives to Safely Unlock >80% Cross-Instrument Library Recall

In Phase 1, we discovered that lowering the library matching threshold to `tau >= 0.40` unlocks **81.5% Recall** (up from 2.1% at `tau >= 0.90`), but introduces a **3.35% Decoy False Positive rate**.

In this Phase 2 experiment, we:
1. Conduct an empirical autopsy on the decoys scoring `>= 0.40` to determine the exact failure modes (1-peak wonders, low-m/z motif traps, precursor dominance).
2. Formulate and test **Smart Multi-Dimensional Filters**:
   - `Filter A`: Precursor peak masking (`|mz - M_prec| > 1.5 Da`)
   - `Filter B`: Minimum matched fragment count (`N_matched >= 2, 3, 4`)
   - `Filter C`: Core scaffold diagnostic fragment (`max_matched_mz > 0.5 * M_prec`)
   - `Filter D`: Combined Smart Gate
3. Measure whether the Smart Gate suppresses Decoy False Positives while preserving **>80% True Match Recall**.""")

add_code("""import sys, os, time, json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.dpi": 150, "axes.titleweight": "bold"})

def locate_data_dir():
    candidates = [
        Path("/kaggle/input/competitions/enveda-CASMI26-molecule-id-mass-spectra"),
        Path("/kaggle/input/enveda-casmi26-molecule-id-mass-spectra"),
        Path("/kaggle/input/enveda-CASMI26-molecule-id-mass-spectra"),
        Path("data/raw"),
        Path("../data/raw"),
    ]
    for p in candidates:
        if (p / "train.parquet").exists():
            return p.resolve()
    raise FileNotFoundError("Could not find train.parquet")

DATA_DIR = locate_data_dir()
TRAIN_PATH = DATA_DIR / "train.parquet"
print(f"Dataset path resolved: {TRAIN_PATH}")
""")

add_code("""print("Reading metadata from train.parquet...")
t0 = time.time()
cols = ["ingest_lib", "inchikey14", "adduct", "precursor_mz", "ionization_mode", "instrument_type", "num_peaks"]
meta = pd.read_parquet(TRAIN_PATH, columns=cols)
meta["row_id"] = np.arange(len(meta), dtype=np.int64)
print(f"Loaded {len(meta):,} rows in {time.time()-t0:.1f}s.")

# Domain A: timsTOF (Enveda)
timstof_libs = {"enveda-180", "enveda-np-examples"}
is_timstof = meta["ingest_lib"].isin(timstof_libs)

df_timstof = meta[is_timstof]
df_ref = meta[~is_timstof]

common_keys = sorted(set(df_timstof["inchikey14"]).intersection(set(df_ref["inchikey14"])))
print(f"Total timsTOF spectra: {len(df_timstof):,} across {df_timstof['inchikey14'].nunique():,} structures")
print(f"Total reference spectra: {len(df_ref):,} across {df_ref['inchikey14'].nunique():,} structures")
print(f"Shared structures (Cross-Instrument Overlap): {len(common_keys):,}")
""")

add_code("""# Sample 3,000 structures for high statistical power
SAMPLE_SIZE = 3000
rng = np.random.default_rng(20261011)
sampled_keys = rng.choice(common_keys, size=min(SAMPLE_SIZE, len(common_keys)), replace=False)

sub_timstof = df_timstof[df_timstof["inchikey14"].isin(sampled_keys)]
sub_ref = df_ref[df_ref["inchikey14"].isin(sampled_keys)]

tp_pairs = []
decoy_pairs = []

# Index ref by precursor mass for fast isobaric decoy lookup (+-10 ppm)
ref_mz = df_ref["precursor_mz"].to_numpy()
ref_row = df_ref["row_id"].to_numpy()
ref_keys = df_ref["inchikey14"].to_numpy()
ref_order = np.argsort(ref_mz)
ref_mz_sorted = ref_mz[ref_order]
ref_row_sorted = ref_row[ref_order]
ref_keys_sorted = ref_keys[ref_order]

print("Assembling paired true matches and isobaric decoys...")
for key in sampled_keys:
    t_group = sub_timstof[sub_timstof["inchikey14"] == key]
    r_group = sub_ref[sub_ref["inchikey14"] == key]
    
    common_adducts = set(t_group["adduct"]).intersection(set(r_group["adduct"]))
    if common_adducts:
        ad = list(common_adducts)[0]
        t_row = t_group[t_group["adduct"] == ad].iloc[0]
        r_row = r_group[r_group["adduct"] == ad].iloc[0]
    else:
        t_row = t_group.iloc[0]
        r_row = r_group.iloc[0]
        
    t_id = int(t_row["row_id"])
    r_id = int(r_row["row_id"])
    pmz = float(t_row["precursor_mz"])
    
    tp_pairs.append({
        "t_id": t_id, "r_id": r_id, "key": key,
        "adduct_t": t_row["adduct"], "adduct_r": r_row["adduct"],
        "lib_t": t_row["ingest_lib"], "lib_r": r_row["ingest_lib"],
        "mz": pmz
    })
    
    # Decoy pairing within +-10 ppm with different InChIKey14
    tol = pmz * 10e-6
    left = np.searchsorted(ref_mz_sorted, pmz - tol, side='left')
    right = np.searchsorted(ref_mz_sorted, pmz + tol, side='right')
    
    if right > left:
        cand_idx = np.where(ref_keys_sorted[left:right] != key)[0]
        if len(cand_idx) > 0:
            d_pick = left + cand_idx[0]
            decoy_pairs.append({
                "t_id": t_id, "d_id": int(ref_row_sorted[d_pick]), 
                "key_t": key, "key_d": ref_keys_sorted[d_pick],
                "mz": pmz
            })

print(f"Total True Positive pairs: {len(tp_pairs):,}")
print(f"Total Isobaric Decoy pairs: {len(decoy_pairs):,}")
""")

add_code("""all_row_ids = sorted(set([p["t_id"] for p in tp_pairs] + 
                         [p["r_id"] for p in tp_pairs] + 
                         [p["d_id"] for p in decoy_pairs]))
print(f"Reading {len(all_row_ids):,} spectral rows from parquet...")

pf = pq.ParquetFile(TRAIN_PATH)
row_map = {}
needed_set = set(all_row_ids)

start_row = 0
for rg in range(pf.metadata.num_row_groups):
    n_rows = pf.metadata.row_group(rg).num_rows
    end_row = start_row + n_rows
    local_ids = [rid for rid in needed_set if start_row <= rid < end_row]
    if local_ids:
        local_offsets = [rid - start_row for rid in local_ids]
        table = pf.read_row_group(rg, columns=["ms2_mzs", "ms2_normalized_intensities"]).take(pa.array(local_offsets))
        mzs = table["ms2_mzs"].to_pylist()
        ints = table["ms2_normalized_intensities"].to_pylist()
        for rid, mz, it in zip(local_ids, mzs, ints):
            row_map[rid] = (np.asarray(mz, dtype=np.float64), np.asarray(it, dtype=np.float64))
    start_row = end_row

print(f"Successfully loaded {len(row_map):,} spectra into memory.")
""")

add_code("""def compute_spectral_diagnostics(mzs1, ints1, mzs2, ints2, prec_mz, mz_tol_ppm=15.0, power=0.5):
    \"\"\"
    Computes spectral matching with granular diagnostic metrics:
    1. cos_sqrt: baseline square-root weighted cosine
    2. cos_masked: cosine excluding peaks within 1.5 Da of precursor
    3. n_matched: number of aligned peaks
    4. has_high_mz: whether any matched peak is > 0.5 * prec_mz (excluding intact precursor)
    5. max_matched_mz: m/z of highest matched peak
    6. explained_frac: fraction of query base-intensity explained by matched peaks
    \"\"\"
    if len(mzs1) == 0 or len(mzs2) == 0:
        return {
            "cos_sqrt": 0.0, "cos_masked": 0.0, "n_matched": 0,
            "has_high_mz": False, "max_matched_mz": 0.0, "explained_frac": 0.0
        }
    
    is_frag1 = np.abs(mzs1 - prec_mz) > 1.5
    is_frag2 = np.abs(mzs2 - prec_mz) > 1.5
    
    w1 = np.power(ints1, power)
    w2 = np.power(ints2, power)
    norm1 = np.linalg.norm(w1)
    norm2 = np.linalg.norm(w2)
    
    w1_m = np.where(is_frag1, w1, 0.0)
    w2_m = np.where(is_frag2, w2, 0.0)
    norm1_m = np.linalg.norm(w1_m)
    norm2_m = np.linalg.norm(w2_m)
    
    dot = 0.0
    dot_masked = 0.0
    n_matched = 0
    matched_mzs = []
    explained_intensity = 0.0
    
    j_start = 0
    n2 = len(mzs2)
    half_prec = 0.5 * prec_mz
    
    for i, (m1, val1) in enumerate(zip(mzs1, w1)):
        tol = m1 * mz_tol_ppm * 1e-6
        best_val2 = 0.0
        best_idx2 = -1
        best_diff = tol
        for j in range(j_start, n2):
            m2 = mzs2[j]
            diff = abs(m1 - m2)
            if m2 < m1 - tol:
                j_start = j
                continue
            if m2 > m1 + tol:
                break
            if diff <= best_diff:
                best_diff = diff
                best_val2 = w2[j]
                best_idx2 = j
        
        if best_val2 > 0:
            n_matched += 1
            matched_mzs.append(m1)
            dot += val1 * best_val2
            explained_intensity += ints1[i]
            if is_frag1[i] and is_frag2[best_idx2]:
                dot_masked += w1_m[i] * w2_m[best_idx2]
                
    cos_sqrt = float(dot / (norm1 * norm2)) if (norm1 > 0 and norm2 > 0) else 0.0
    cos_masked = float(dot_masked / (norm1_m * norm2_m)) if (norm1_m > 0 and norm2_m > 0) else 0.0
    
    # Check for high-m/z fragment (excluding precursor)
    frag_matches = [m for m in matched_mzs if abs(m - prec_mz) > 1.5]
    has_high_mz = any(m > half_prec for m in frag_matches)
    max_matched_mz = max(frag_matches) if frag_matches else 0.0
    explained_frac = float(explained_intensity / (np.sum(ints1) + 1e-9))
    
    return {
        "cos_sqrt": cos_sqrt,
        "cos_masked": cos_masked,
        "n_matched": n_matched,
        "has_high_mz": has_high_mz,
        "max_matched_mz": max_matched_mz,
        "explained_frac": explained_frac
    }

print("Running spectral diagnostics on True Positives and Decoys...")
tp_metrics = []
for p in tp_pairs:
    mz1, it1 = row_map[p["t_id"]]
    mz2, it2 = row_map[p["r_id"]]
    d = compute_spectral_diagnostics(mz1, it1, mz2, it2, p["mz"], mz_tol_ppm=15.0, power=0.5)
    tp_metrics.append(d)

decoy_metrics = []
for p in decoy_pairs:
    mz1, it1 = row_map[p["t_id"]]
    mz2, it2 = row_map[p["d_id"]]
    d = compute_spectral_diagnostics(mz1, it1, mz2, it2, p["mz"], mz_tol_ppm=15.0, power=0.5)
    decoy_metrics.append(d)

df_tp = pd.DataFrame(tp_metrics)
df_decoy = pd.DataFrame(decoy_metrics)
print("Finished extracting all diagnostic metrics.")
""")

add_code("""print("\\n" + "="*75)
print("PART 1: AUTOPSY OF THE DECOY FALSE POSITIVES (Cosine >= 0.40)")
print("="*75)

high_decoys = df_decoy[df_decoy["cos_sqrt"] >= 0.40]
n_high = len(high_decoys)
print(f"Total Decoys with Cosine >= 0.40: {n_high} ({n_high / len(df_decoy):.2%} of all decoys)")

# Failure Mode 1: Low peak matches (1 or 2 peaks only)
low_peaks = (high_decoys["n_matched"] <= 2).sum()
print(f"1. Low-peak match trap (matched <= 2 peaks): {low_peaks} ({low_peaks/n_high:.1%})")

# Failure Mode 2: Missing high-m/z core fragment
no_high_mz = (~high_decoys["has_high_mz"]).sum()
print(f"2. Low-mass motif trap (zero matched peaks > 0.5 * precursor): {no_high_mz} ({no_high_mz/n_high:.1%})")

# Failure Mode 3: Precursor inflation (masked cosine drops by >= 0.20)
prec_drop = ((high_decoys["cos_sqrt"] - high_decoys["cos_masked"]) >= 0.20).sum()
print(f"3. Intact precursor inflation (score drops >= 0.20 when precursor masked): {prec_drop} ({prec_drop/n_high:.1%})")

# Combined elimination rate on high decoys
eliminated = (
    (high_decoys["n_matched"] <= 2) | 
    (~high_decoys["has_high_mz"]) | 
    (high_decoys["cos_masked"] < 0.35)
).sum()
print(f"\\n--> SMART FILTER EFFECTIVENESS ON DECOYS: Eliminates {eliminated}/{n_high} ({eliminated/n_high:.1%}) of the decoys!")
""")

add_code("""print("\\n" + "="*75)
print("PART 2: BENCHMARKING FILTER ABLATIONS ACROSS THRESHOLDS")
print("="*75)

# Define Filter Rules:
# Baseline: cos_sqrt >= tau
# Filter 1 (Masked): cos_masked >= tau
# Filter 2 (N_matched >= 3): cos_sqrt >= tau & n_matched >= 3
# Filter 3 (N_matched >= 4): cos_sqrt >= tau & n_matched >= 4
# Smart Gate: cos_masked >= tau & n_matched >= 3 & has_high_mz

thresholds = [0.90, 0.80, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40]
results = []

for tau in thresholds:
    # Baseline
    tp_base = (df_tp["cos_sqrt"] >= tau).mean() * 100
    fp_base = (df_decoy["cos_sqrt"] >= tau).mean() * 100
    
    # Filter: N_matched >= 3
    tp_n3 = ((df_tp["cos_sqrt"] >= tau) & (df_tp["n_matched"] >= 3)).mean() * 100
    fp_n3 = ((df_decoy["cos_sqrt"] >= tau) & (df_decoy["n_matched"] >= 3)).mean() * 100
    
    # Filter: N_matched >= 4
    tp_n4 = ((df_tp["cos_sqrt"] >= tau) & (df_tp["n_matched"] >= 4)).mean() * 100
    fp_n4 = ((df_decoy["cos_sqrt"] >= tau) & (df_decoy["n_matched"] >= 4)).mean() * 100
    
    # Combined Smart Gate: cos_masked >= tau & n_matched >= 3 & has_high_mz
    tp_smart = (
        (df_tp["cos_masked"] >= tau) & 
        (df_tp["n_matched"] >= 3) & 
        (df_tp["has_high_mz"])
    ).mean() * 100
    
    fp_smart = (
        (df_decoy["cos_masked"] >= tau) & 
        (df_decoy["n_matched"] >= 3) & 
        (df_decoy["has_high_mz"])
    ).mean() * 100
    
    prec_smart = tp_smart / (tp_smart + fp_smart + 1e-9) * 100
    
    results.append({
        "Threshold (tau)": tau,
        "Base Recall (%)": tp_base,
        "Base Decoy FPR (%)": fp_base,
        "N>=3 Decoy FPR (%)": fp_n3,
        "Smart Recall (%)": tp_smart,
        "Smart Decoy FPR (%)": fp_smart,
        "Smart Precision (%)": prec_smart
    })

res_df = pd.DataFrame(results)
print(res_df.to_string(index=False))

os.makedirs("/kaggle/working", exist_ok=True)
res_df.to_csv("/kaggle/working/smart_gate_ablation_results.csv", index=False)

summary_dict = {
    "num_tp_pairs": len(df_tp),
    "num_decoy_pairs": len(df_decoy),
    "decoy_autopsy": {
        "n_high_decoys": int(n_high),
        "low_peaks_pct": float(low_peaks / n_high),
        "no_high_mz_pct": float(no_high_mz / n_high),
        "precursor_inflation_pct": float(prec_drop / n_high),
        "eliminated_pct": float(eliminated / n_high)
    },
    "ablation_table": res_df.to_dict(orient="records")
}
Path("/kaggle/working/smart_gate_report.json").write_text(json.dumps(summary_dict, indent=2))
print("\\nSaved report to /kaggle/working/smart_gate_report.json")
""")

add_code("""fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

# Panel 1: Decoy Elimination Comparison
axes[0].plot(res_df["Threshold (tau)"], res_df["Base Decoy FPR (%)"], marker="o", color="#d62828", label="Baseline Decoy FPR (No Filter)")
axes[0].plot(res_df["Threshold (tau)"], res_df["N>=3 Decoy FPR (%)"], marker="^", color="#f77f00", label="Filter: N_matched >= 3")
axes[0].plot(res_df["Threshold (tau)"], res_df["Smart Decoy FPR (%)"], marker="s", color="#2a9d8f", linewidth=2.5, label="Smart Gate (N>=3 + High-m/z + Masked)")
axes[0].set_title("Decoy False Positive Suppression by Filter", fontsize=14)
axes[0].set_xlabel("Library Cosine Cutoff (tau)")
axes[0].set_ylabel("Decoy False Positive Rate (%)")
axes[0].legend(loc="upper right")

# Panel 2: True Match Recall Preservation
axes[1].plot(res_df["Threshold (tau)"], res_df["Base Recall (%)"], marker="o", color="#457b9d", linestyle="--", label="Raw Cosine Recall")
axes[1].plot(res_df["Threshold (tau)"], res_df["Smart Recall (%)"], marker="s", color="#2a9d8f", linewidth=2.5, label="Smart Gate Recall")
axes[1].axvline(0.90, color="#d62828", linestyle=":", label="Old Baseline (tau = 0.90: ~2% Recall)")
axes[1].set_title("True Match Recall Preservation", fontsize=14)
axes[1].set_xlabel("Library Cosine Cutoff (tau)")
axes[1].set_ylabel("True Match Recall (%)")
axes[1].legend(loc="lower left")

plt.tight_layout()
plt.savefig("/kaggle/working/smart_gate_decoy_suppression.png", dpi=200)
plt.show()
print("Saved figure to /kaggle/working/smart_gate_decoy_suppression.png")
""")

# Assign cell IDs
for i, c in enumerate(cells):
    c["id"] = f"cell_{i}"

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

target_file = Path(".scratch/kaggle_libmax_eval/casmi_cross_instrument_libmax_eval.ipynb")
target_file.write_text(json.dumps(nb, indent=2), encoding="utf-8")
print(f"Generated Phase 2 Notebook: {target_file} with {len(cells)} cells.")
