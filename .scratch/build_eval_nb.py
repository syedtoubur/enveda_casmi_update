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

add_md("""# CASMI 2026: Cross-Instrument Library Matching Statistical Evaluation
### Empirical Testing of `lib_max >= 0.90` Hard Constraint on timsTOF vs Non-timsTOF Identical Molecules

This notebook empirically evaluates mass-spectral cosine similarity for identical molecules across different instrument domains in the CASMI 2026 corpus:
1. **Target Instrument**: Bruker timsTOF (`enveda-180` and `enveda-np-examples`).
2. **Reference Libraries**: Orbitrap, Q-TOF, FT-ICR (`gnps`, `massbank`, `mona`, `riken`, `pluskal_ms2`, `msdial`).

We test whether the hardcoded baseline constraint `lib_max >= 0.90` holds when a query matches an identical molecule measured on a different instrument family.""")

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
print(f"Shared structures (Ground Truth Cross-Instrument Overlap): {len(common_keys):,}")
""")

add_code("""# Sample representative structures
SAMPLE_SIZE = 2500
rng = np.random.default_rng(20261011)
sampled_keys = rng.choice(common_keys, size=min(SAMPLE_SIZE, len(common_keys)), replace=False)

sub_timstof = df_timstof[df_timstof["inchikey14"].isin(sampled_keys)]
sub_ref = df_ref[df_ref["inchikey14"].isin(sampled_keys)]

tp_pairs = []
decoy_pairs = []

# Index ref by precursor mass for fast decoy lookup
ref_mz = df_ref["precursor_mz"].to_numpy()
ref_row = df_ref["row_id"].to_numpy()
ref_keys = df_ref["inchikey14"].to_numpy()
ref_order = np.argsort(ref_mz)
ref_mz_sorted = ref_mz[ref_order]
ref_row_sorted = ref_row[ref_order]
ref_keys_sorted = ref_keys[ref_order]

print("Pairing identical molecules and isobaric decoys...")
for key in sampled_keys:
    t_group = sub_timstof[sub_timstof["inchikey14"] == key]
    r_group = sub_ref[sub_ref["inchikey14"] == key]
    
    # Try to match same adduct
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
    tp_pairs.append({
        "t_id": t_id, "r_id": r_id, "key": key,
        "adduct_t": t_row["adduct"], "adduct_r": r_row["adduct"],
        "lib_t": t_row["ingest_lib"], "lib_r": r_row["ingest_lib"],
        "mz": t_row["precursor_mz"]
    })
    
    # Decoy: within +-10 ppm, different molecule
    pmz = t_row["precursor_mz"]
    tol = pmz * 10e-6
    left = np.searchsorted(ref_mz_sorted, pmz - tol, side='left')
    right = np.searchsorted(ref_mz_sorted, pmz + tol, side='right')
    
    if right > left:
        cand_idx = np.where(ref_keys_sorted[left:right] != key)[0]
        if len(cand_idx) > 0:
            d_pick = left + cand_idx[0]
            decoy_pairs.append({
                "t_id": t_id, "d_id": int(ref_row_sorted[d_pick]), "key_t": key, "key_d": ref_keys_sorted[d_pick],
                "mz": pmz
            })

print(f"Assembled {len(tp_pairs):,} True Positive pairs and {len(decoy_pairs):,} Isobaric Decoy pairs.")
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

add_code("""def compute_cosine(mzs1, ints1, mzs2, ints2, mz_tol_ppm=15.0, power=0.5):
    if len(mzs1) == 0 or len(mzs2) == 0:
        return 0.0
    w1 = np.power(ints1, power)
    w2 = np.power(ints2, power)
    norm1 = np.linalg.norm(w1)
    norm2 = np.linalg.norm(w2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    dot = 0.0
    j_start = 0
    n2 = len(mzs2)
    for m1, val1 in zip(mzs1, w1):
        tol = m1 * mz_tol_ppm * 1e-6
        best_val2 = 0.0
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
                best_val2 = max(best_val2, w2[j])
        if best_val2 > 0:
            dot += val1 * best_val2
    return float(dot / (norm1 * norm2))

print("Computing spectral similarities...")
tp_cosine_sqrt = []
tp_cosine_raw = []
decoy_cosine_sqrt = []
decoy_cosine_raw = []

for p in tp_pairs:
    mz1, it1 = row_map[p["t_id"]]
    mz2, it2 = row_map[p["r_id"]]
    tp_cosine_sqrt.append(compute_cosine(mz1, it1, mz2, it2, mz_tol_ppm=15.0, power=0.5))
    tp_cosine_raw.append(compute_cosine(mz1, it1, mz2, it2, mz_tol_ppm=15.0, power=1.0))

for p in decoy_pairs:
    mz1, it1 = row_map[p["t_id"]]
    mz2, it2 = row_map[p["d_id"]]
    decoy_cosine_sqrt.append(compute_cosine(mz1, it1, mz2, it2, mz_tol_ppm=15.0, power=0.5))
    decoy_cosine_raw.append(compute_cosine(mz1, it1, mz2, it2, mz_tol_ppm=15.0, power=1.0))

tp_sqrt = np.array(tp_cosine_sqrt)
tp_raw = np.array(tp_cosine_raw)
decoy_sqrt = np.array(decoy_cosine_sqrt)
decoy_raw = np.array(decoy_cosine_raw)
print(f"Finished scoring {len(tp_sqrt):,} True Positives and {len(decoy_sqrt):,} Decoys.")
""")

add_code("""quantiles = [5, 10, 25, 50, 75, 90, 95]
q_df = pd.DataFrame({
    "Percentile": [f"{q}%" for q in quantiles],
    "True Match (Sqrt Cosine)": [np.percentile(tp_sqrt, q) for q in quantiles],
    "True Match (Raw Cosine)": [np.percentile(tp_raw, q) for q in quantiles],
    "Decoy (Sqrt Cosine)": [np.percentile(decoy_sqrt, q) for q in quantiles],
    "Decoy (Raw Cosine)": [np.percentile(decoy_raw, q) for q in quantiles],
})
print("\\n" + "="*75)
print("TABLE 1. DISTRIBUTION OF CROSS-INSTRUMENT SIMILARITY")
print("="*75)
print(q_df.to_string(index=False))

thresholds = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.50, 0.40]
thresh_rows = []
for tau in thresholds:
    tp_rec = np.mean(tp_sqrt >= tau) * 100
    fp_rate = np.mean(decoy_sqrt >= tau) * 100
    prec = tp_rec / (tp_rec + fp_rate + 1e-9) * 100
    thresh_rows.append({
        "Threshold (tau)": tau,
        "True Match Recall (%)": tp_rec,
        "Decoy False Positive (%)": fp_rate,
        "Precision Estimate (%)": prec
    })
t_df = pd.DataFrame(thresh_rows)
print("\\n" + "="*75)
print("TABLE 2. SENSITIVITY AND DECOY CONTAMINATION BY THRESHOLD (Sqrt Cosine)")
print("="*75)
print(t_df.to_string(index=False))

pass_09 = float(np.mean(tp_sqrt >= 0.90) * 100)
report = {
    "num_tp_pairs": len(tp_sqrt),
    "num_decoy_pairs": len(decoy_sqrt),
    "recall_at_0_90": pass_09,
    "false_rejection_at_0_90": 100.0 - pass_09,
    "median_true_match_sim": float(np.median(tp_sqrt)),
    "mean_true_match_sim": float(np.mean(tp_sqrt)),
    "table_quantiles": q_df.to_dict(orient="records"),
    "table_thresholds": t_df.to_dict(orient="records")
}
os.makedirs("/kaggle/working", exist_ok=True)
Path("/kaggle/working/cross_instrument_libmax_report.json").write_text(json.dumps(report, indent=2))
t_df.to_csv("/kaggle/working/summary_threshold_metrics.csv", index=False)
print(f"\\nSaved report to /kaggle/working/cross_instrument_libmax_report.json")
""")

add_code("""fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Panel 1: Distributions
sns.kdeplot(tp_sqrt, ax=axes[0], label="True Same-Molecule Cross-Instrument", color="#2a9d8f", fill=True, bw_adjust=0.8)
sns.kdeplot(decoy_sqrt, ax=axes[0], label="Isobaric Decoys (+-10 ppm)", color="#e76f51", fill=True, bw_adjust=0.8)
axes[0].axvline(0.90, color="#d62828", linestyle="--", linewidth=2, label="Hard Gate Cutoff (tau = 0.90)")
axes[0].set_title("Cross-Instrument Spectral Cosine Distribution", fontsize=14)
axes[0].set_xlabel("Peak Cosine Similarity (Sqrt Intensity)")
axes[0].set_ylabel("Density")
axes[0].legend(loc="upper left")

# Panel 2: ROC / Recall Curve
axes[1].plot(t_df["Threshold (tau)"], t_df["True Match Recall (%)"], marker="o", color="#2a9d8f", label="True Match Recall")
axes[1].plot(t_df["Threshold (tau)"], t_df["Decoy False Positive (%)"], marker="s", color="#e76f51", label="Decoy False Positives")
axes[1].axvline(0.90, color="#d62828", linestyle="--", label="tau = 0.90")
axes[1].set_title("Recall vs Decoy FPR as a Function of tau", fontsize=14)
axes[1].set_xlabel("Library Similarity Cutoff (tau)")
axes[1].set_ylabel("Percentage (%)")
axes[1].legend(loc="center left")

plt.tight_layout()
plt.savefig("/kaggle/working/cross_instrument_libmax_distributions.png", dpi=200)
plt.show()
print("Figure saved to /kaggle/working/cross_instrument_libmax_distributions.png")
""")

# Assign valid unique IDs to each cell
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
print(f"Generated {target_file} with {len(cells)} cells and valid kernelspec.")

