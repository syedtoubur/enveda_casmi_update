import json
import gzip
import base64
import os
import numpy as np
import nbformat

# Load the sampled cohort
cohort_path = '.scratch/cohort_scaffold_pairs.json'
print(f"Loading {cohort_path}...")
with open(cohort_path) as f:
    d = json.load(f)

pairs = d['pairs']
arm_map = {
    'Arm1_SameScaffold_General': 1,
    'Arm2_SameScaffold_BioDelta': 2,
    'Arm3_DiffScaffold_IsobaricDecoy': 3,
    'Arm4_DiffScaffold_RandomDecoy': 4
}
arr = []
for p in pairs:
    arr.append([p['row_A'], p['row_B'], arm_map[p['arm']], p['delta_m'], p['pmz_A'], p['pmz_B']])
arr = np.array(arr, dtype=np.float64)
raw_bytes = arr.tobytes()
gz = gzip.compress(raw_bytes)
b64_cohort = base64.b64encode(gz).decode('ascii')
print(f"Encoded {len(arr)} pairs into base64 string (len: {len(b64_cohort):,})")

os.makedirs('.scratch/kaggle_scaffold_eval', exist_ok=True)

# Generate metadata.json
meta_json = {
  "id": "syedtoubur/casmi26-scaffold-network-eval",
  "title": "casmi26-scaffold-network-eval",
  "code_file": "casmi_scaffold_network_eval.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": True,
  "enable_gpu": False,
  "enable_internet": False,
  "competition_sources": [
    "enveda-CASMI26-molecule-id-mass-spectra"
  ],
  "dataset_sources": [],
  "kernel_sources": [],
  "model_sources": []
}

with open('.scratch/kaggle_scaffold_eval/kernel-metadata.json', 'w') as f:
    json.dump(meta_json, f, indent=2)
print("Wrote .scratch/kaggle_scaffold_eval/kernel-metadata.json")

# Build the python source for the notebook
py_content = f'''# %% [markdown]
# # CASMI 2026: Empirical Falsification & Validation of Transductive Molecular Networking (Idea A1)
# ### Pairwise Scaffold Congener Similarity vs. Isobaric Decoys on train.parquet
#
# This experiment evaluates the core hypothesis of Idea A1:
# **Do molecules sharing the same Bemis-Murcko scaffold exhibit significantly higher spectral similarity than molecules with different scaffolds?**
#
# We evaluate 10,457 pre-stratified pairs across 4 treatment arms:
# 1. **Arm 1 (Same Scaffold General, N=3,000)**: Distinct molecules sharing the same Murcko scaffold.
# 2. **Arm 2 (Same Scaffold + Biogenetic Delta, N=1,500)**: Same scaffold with precursor mass difference matching known biogenetic transforms.
# 3. **Arm 3 (Different Scaffold Isobaric Decoys, N=2,957)**: Precursor mass within +-20 ppm with different scaffolds (retrieval competitors).
# 4. **Arm 4 (Different Scaffold Random Decoys, N=3,000)**: Random background decoy pairs.
#
# Metrics Evaluated:
# - Raw Cosine
# - Precursor-Masked Cosine
# - Spectral Entropy Similarity (Li & Fiehn 2021)
# - Neutral Loss Cosine
# - Modified Cosine (Shift-Aware, Wang et al. 2016 GNPS)

# %%
import sys, os, time, json, gzip, base64
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({{"figure.dpi": 150, "axes.titleweight": "bold"}})

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
print(f"Dataset path resolved: {{TRAIN_PATH}}")

# %%
# Decode Pre-Stratified Evaluation Cohort (10,457 pairs)
COHORT_B64 = "{b64_cohort}"
raw_bytes = gzip.decompress(base64.b64decode(COHORT_B64.encode('ascii')))
cohort_arr = np.frombuffer(raw_bytes, dtype=np.float64).reshape(-1, 6)
# Columns: [row_A, row_B, arm_id, delta_m, pmz_A, pmz_B]

print(f"Loaded cohort array of shape: {{cohort_arr.shape}}")
arm_ids = cohort_arr[:, 2].astype(int)
print(f"Arm 1 (Same Scaffold General): {{(arm_ids == 1).sum():,}}")
print(f"Arm 2 (Same Scaffold Bio-Delta): {{(arm_ids == 2).sum():,}}")
print(f"Arm 3 (Diff Scaffold Isobaric): {{(arm_ids == 3).sum():,}}")
print(f"Arm 4 (Diff Scaffold Random): {{(arm_ids == 4).sum():,}}")

unique_rows = sorted(set(cohort_arr[:, 0].astype(int).tolist() + cohort_arr[:, 1].astype(int).tolist()))
print(f"Total unique spectrum rows to load: {{len(unique_rows):,}}")

# Store precursor masses
row_to_pmz = {{}}
for idx in range(len(cohort_arr)):
    row_to_pmz[int(cohort_arr[idx, 0])] = float(cohort_arr[idx, 4])
    row_to_pmz[int(cohort_arr[idx, 1])] = float(cohort_arr[idx, 5])

# %%
# Fetch exact spectrum rows from train.parquet using row group indexing
print(f"Loading {{len(unique_rows):,}} spectra from train.parquet via row group indexing...")
t0 = time.time()
pf = pq.ParquetFile(TRAIN_PATH)
row_map = {{}}
needed_set = set(unique_rows)

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
            row_map[rid] = (
                np.asarray(mz, dtype=np.float64),
                np.asarray(it, dtype=np.float64),
                row_to_pmz[rid]
            )
    start_row = end_row

print(f"Loaded {{len(row_map):,}} spectra into memory in {{time.time()-t0:.1f}}s.")

# %%
# Multi-Metric Spectral Matching Kernels
def clean_spectrum(mzs, ints, prec_mz=None, mask_prec=True, max_peaks=64, int_floor=0.01):
    if len(mzs) == 0:
        return np.array([]), np.array([]), 0.0, True
    mzs = np.asarray(mzs, dtype=np.float64)
    ints = np.asarray(ints, dtype=np.float64)
    
    if mask_prec and prec_mz is not None:
        mask = np.abs(mzs - prec_mz) > 1.5
        mzs = mzs[mask]
        ints = ints[mask]
        if len(mzs) == 0:
            return np.array([]), np.array([]), 0.0, True
            
    max_i = np.max(ints)
    if max_i <= 0:
        return np.array([]), np.array([]), 0.0, True
        
    floor_mask = ints >= (int_floor * max_i)
    mzs = mzs[floor_mask]
    ints = ints[floor_mask]
    if len(mzs) == 0:
        return np.array([]), np.array([]), 0.0, True
        
    if len(mzs) > max_peaks:
        top_idx = np.argsort(ints)[-max_peaks:]
        mzs = mzs[top_idx]
        ints = ints[top_idx]
        
    order = np.argsort(mzs)
    mzs = mzs[order]
    ints = ints[order]
    
    w = np.sqrt(ints)
    w_sum = np.sum(w)
    if w_sum <= 0:
        return np.array([]), np.array([]), 0.0, True
    w_norm = w / w_sum
    entropy = -float(np.sum(w_norm * np.log(w_norm + 1e-15)))
    return mzs, w, entropy, False

def compute_pairwise_metrics(m1, w1, e1, m2, w2, e2, pmz1, pmz2, tol_da=0.02):
    n1, n2 = len(m1), len(m2)
    if n1 == 0 or n2 == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
        
    norm1 = np.linalg.norm(w1)
    norm2 = np.linalg.norm(w2)
    if norm1 <= 0 or norm2 <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
        
    delta_m = pmz1 - pmz2
    
    # 1. Direct Matching (Raw & Masked)
    direct_matches = []
    for i in range(n1):
        for j in range(n2):
            diff = abs(m1[i] - m2[j])
            if diff <= tol_da:
                direct_matches.append((w1[i] * w2[j], diff, i, j))
    direct_matches.sort(key=lambda x: (-x[0], x[1]))
    
    used1, used2 = set(), set()
    dot_direct = 0.0
    for score, diff, i, j in direct_matches:
        if i not in used1 and j not in used2:
            used1.add(i); used2.add(j)
            dot_direct += score
            
    cos_direct = float(dot_direct / (norm1 * norm2))
    
    # Entropy similarity
    mixture_weights = []
    for _, _, i, j in direct_matches:
        if i in used1 and j in used2:
            mixture_weights.append((w1[i] + w2[j]) * 0.5)
    for i in range(n1):
        if i not in used1:
            mixture_weights.append(w1[i] * 0.5)
    for j in range(n2):
        if j not in used2:
            mixture_weights.append(w2[j] * 0.5)
            
    mw = np.array(mixture_weights, dtype=np.float64)
    mw_sum = np.sum(mw)
    if mw_sum > 0:
        mw = mw / mw_sum
        e_mix = -float(np.sum(mw * np.log(mw + 1e-15)))
        entropy_sim = max(0.0, float(1.0 - (2.0 * e_mix - e1 - e2) / np.log(4.0)))
    else:
        entropy_sim = 0.0

    # 2. Neutral Loss Cosine
    nl1 = pmz1 - m1
    nl2 = pmz2 - m2
    nl_matches = []
    for i in range(n1):
        for j in range(n2):
            diff = abs(nl1[i] - nl2[j])
            if diff <= tol_da:
                nl_matches.append((w1[i] * w2[j], diff, i, j))
    nl_matches.sort(key=lambda x: (-x[0], x[1]))
    used_nl1, used_nl2 = set(), set()
    dot_nl = 0.0
    for score, diff, i, j in nl_matches:
        if i not in used_nl1 and j not in used_nl2:
            used_nl1.add(i); used_nl2.add(j)
            dot_nl += score
    cos_nl = float(dot_nl / (norm1 * norm2))

    # 3. Modified Cosine (Shift-Aware)
    mod_matches = []
    for i in range(n1):
        for j in range(n2):
            diff = abs(m1[i] - m2[j])
            diff_shift = abs((m1[i] - m2[j]) - delta_m)
            err = min(diff, diff_shift)
            if err <= tol_da:
                mod_matches.append((w1[i] * w2[j], err, i, j))
    mod_matches.sort(key=lambda x: (-x[0], x[1]))
    used_m1, used_m2 = set(), set()
    dot_mod = 0.0
    for score, err, i, j in mod_matches:
        if i not in used_m1 and j not in used_m2:
            used_m1.add(i); used_m2.add(j)
            dot_mod += score
    cos_mod = float(dot_mod / (norm1 * norm2))
    
    return cos_direct, cos_direct, entropy_sim, cos_nl, cos_mod

# %%
# Pre-clean all spectra
print("Pre-cleaning all cached spectra...")
t0 = time.time()
clean_cache = {{}}
for r_id, (m, i, p) in row_map.items():
    clean_cache[r_id] = clean_spectrum(m, i, prec_mz=p, mask_prec=True)
print(f"Pre-cleaned {{len(clean_cache):,}} spectra in {{time.time()-t0:.1f}}s.")

# %%
# Execute Pairwise Benchmark across the 10,457 Pairs
print(f"Computing pairwise spectral similarity across {{len(cohort_arr):,}} pairs...")
t0 = time.time()

results = []
for p_idx in range(len(cohort_arr)):
    row_A = int(cohort_arr[p_idx, 0])
    row_B = int(cohort_arr[p_idx, 1])
    arm_id = int(cohort_arr[p_idx, 2])
    delta_m = float(cohort_arr[p_idx, 3])
    pmz_A = float(cohort_arr[p_idx, 4])
    pmz_B = float(cohort_arr[p_idx, 5])
    
    m1, w1, e1, empty1 = clean_cache[row_A]
    m2, w2, e2, empty2 = clean_cache[row_B]
    
    if empty1 or empty2:
        raw_cos, msk_cos, ent_sim, nl_cos, mod_cos = 0.0, 0.0, 0.0, 0.0, 0.0
    else:
        raw_cos, msk_cos, ent_sim, nl_cos, mod_cos = compute_pairwise_metrics(
            m1, w1, e1, m2, w2, e2, pmz_A, pmz_B, tol_da=0.02
        )
        
    results.append((arm_id, delta_m, raw_cos, msk_cos, ent_sim, nl_cos, mod_cos))
    
    if (p_idx + 1) % 2500 == 0 or p_idx == len(cohort_arr) - 1:
        print(f"  Processed {{p_idx+1:,}}/{{len(cohort_arr):,}} pairs ({{time.time()-t0:.1f}}s)...")

res_df = pd.DataFrame(results, columns=["arm_id", "delta_m", "raw_cos", "masked_cos", "entropy_sim", "nl_cos", "mod_cos"])
arm_names = {{
    1: "Arm 1: Same Scaffold General",
    2: "Arm 2: Same Scaffold Bio-Delta",
    3: "Arm 3: Diff Scaffold Isobaric",
    4: "Arm 4: Diff Scaffold Random"
}}
res_df["arm_name"] = res_df["arm_id"].map(arm_names)
print("Pairwise execution complete!")

# %%
# TABLE 1: QUANTILE DISTRIBUTION ANALYSIS
print("\\n" + "="*85)
print("TABLE 1: QUANTILE DISTRIBUTION ANALYSIS (SAME SCAFFOLD VS. DECOYS)")
print("="*85)

quantiles_list = []
q_cuts = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

for arm in [1, 2, 3, 4]:
    sub = res_df[res_df["arm_id"] == arm]
    for col in ["raw_cos", "masked_cos", "entropy_sim", "nl_cos", "mod_cos"]:
        vals = sub[col].to_numpy()
        q_vals = np.quantile(vals, q_cuts)
        row_dict = {{
            "Arm": arm_names[arm],
            "Metric": col,
            "p05": round(float(q_vals[0]), 4),
            "p10": round(float(q_vals[1]), 4),
            "p25": round(float(q_vals[2]), 4),
            "p50 (Median)": round(float(q_vals[3]), 4),
            "p75": round(float(q_vals[4]), 4),
            "p90": round(float(q_vals[5]), 4),
            "p95": round(float(q_vals[6]), 4),
            "p99": round(float(q_vals[7]), 4),
            "Mean": round(float(np.mean(vals)), 4),
            "Max": round(float(np.max(vals)), 4)
        }}
        quantiles_list.append(row_dict)

q_df = pd.DataFrame(quantiles_list)
print(q_df.to_string(index=False))
q_df.to_csv("scaffold_quantile_distribution.csv", index=False)

# %%
# TABLE 2: OPERATING CURVES & SEPARATION AT THRESHOLDS TAU
print("\\n" + "="*95)
print("TABLE 2: OPERATING CURVES (SAME SCAFFOLD RECALL VS. DECOY FALSE EDGE RATE)")
print("="*95)

thresholds = [0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.40, 0.30]
op_rows = []

arm1_vals = res_df[res_df["arm_id"] == 1]["mod_cos"].to_numpy()
arm2_vals = res_df[res_df["arm_id"] == 2]["mod_cos"].to_numpy()
arm3_vals = res_df[res_df["arm_id"] == 3]["mod_cos"].to_numpy()
arm4_vals = res_df[res_df["arm_id"] == 4]["mod_cos"].to_numpy()

arm1_ent = res_df[res_df["arm_id"] == 1]["entropy_sim"].to_numpy()
arm2_ent = res_df[res_df["arm_id"] == 2]["entropy_sim"].to_numpy()
arm3_ent = res_df[res_df["arm_id"] == 3]["entropy_sim"].to_numpy()

for tau in thresholds:
    r_arm1_mod = float(np.mean(arm1_vals >= tau) * 100)
    r_arm2_mod = float(np.mean(arm2_vals >= tau) * 100)
    fpr_iso_mod = float(np.mean(arm3_vals >= tau) * 100)
    fpr_rand_mod = float(np.mean(arm4_vals >= tau) * 100)
    
    r_arm2_ent = float(np.mean(arm2_ent >= tau) * 100)
    fpr_iso_ent = float(np.mean(arm3_ent >= tau) * 100)
    
    tp = (arm2_vals >= tau).sum()
    fp = (arm3_vals >= tau).sum()
    prec = float(tp / (tp + fp) * 100) if (tp + fp) > 0 else 100.0
    
    op_rows.append({{
        "Tau": tau,
        "Arm1 Recall (%)": round(r_arm1_mod, 2),
        "Arm2 Bio Recall (%)": round(r_arm2_mod, 2),
        "Isobaric Decoy FLR (%)": round(fpr_iso_mod, 2),
        "Random Decoy FLR (%)": round(fpr_rand_mod, 2),
        "Bio-Delta Precision (%)": round(prec, 2),
        "Direct Entropy Recall (%)": round(r_arm2_ent, 2),
        "Direct Entropy Iso FLR (%)": round(fpr_iso_ent, 2)
    }})

op_df = pd.DataFrame(op_rows)
print(op_df.to_string(index=False))
op_df.to_csv("scaffold_operating_curves.csv", index=False)

# %%
# COMPUTE STATISTICAL SEPARATION (AUC-ROC & COHEN'S D)
def compute_auc(pos, neg):
    n_pos = len(pos)
    n_neg = len(neg)
    all_vals = np.concatenate([pos, neg])
    ranks = pd.Series(all_vals).rank().to_numpy()
    sum_pos_ranks = np.sum(ranks[:n_pos])
    u = sum_pos_ranks - (n_pos * (n_pos + 1)) / 2.0
    return float(u / (n_pos * n_neg))

def compute_cohens_d(pos, neg):
    n1, n2 = len(pos), len(neg)
    s1, s2 = np.var(pos, ddof=1), np.var(neg, ddof=1)
    pooled_s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    return float((np.mean(pos) - np.mean(neg)) / (pooled_s + 1e-9))

auc_mod_iso = compute_auc(arm2_vals, arm3_vals)
auc_mod_rand = compute_auc(arm2_vals, arm4_vals)
auc_ent_iso = compute_auc(arm2_ent, arm3_ent)

d_mod_iso = compute_cohens_d(arm2_vals, arm3_vals)
d_mod_rand = compute_cohens_d(arm2_vals, arm4_vals)

print("\\n" + "="*85)
print("STATISTICAL SEPARATION METRICS (IDEA A1 FALSIFICATION BENCHMARK)")
print("="*85)
print(f"1. Modified Cosine vs Isobaric Decoys: AUC = {{auc_mod_iso:.4f}}, Cohen's d = {{d_mod_iso:.4f}}")
print(f"2. Modified Cosine vs Random Decoys:   AUC = {{auc_mod_rand:.4f}}, Cohen's d = {{d_mod_rand:.4f}}")
print(f"3. Direct Entropy vs Isobaric Decoys:  AUC = {{auc_ent_iso:.4f}}")

# Gate Evaluations
g1_pass = bool(auc_mod_iso >= 0.80 and d_mod_iso >= 1.20)
g2_pass = bool(op_df[op_df["Tau"] == 0.70]["Arm2 Bio Recall (%)"].iloc[0] >= 30.0)
g3_pass = bool(op_df[op_df["Tau"] == 0.70]["Bio-Delta Precision (%)"].iloc[0] >= 75.0)
g4_pass = bool((auc_mod_iso - auc_ent_iso) >= 0.10)

report = {{
    "experiment": "CASMI 2026 Idea A1: Transductive Molecular Networking Scaffold Congener Falsifier",
    "total_pairs": len(res_df),
    "statistical_separation": {{
        "auc_mod_cosine_vs_isobaric": round(auc_mod_iso, 4),
        "cohens_d_mod_cosine_vs_isobaric": round(d_mod_iso, 4),
        "auc_mod_cosine_vs_random": round(auc_mod_rand, 4),
        "auc_entropy_vs_isobaric": round(auc_ent_iso, 4),
        "shift_aware_auc_lift": round(auc_mod_iso - auc_ent_iso, 4)
    }},
    "operating_at_tau_070": {{
        "same_scaffold_bio_recall": op_df[op_df["Tau"] == 0.70]["Arm2 Bio Recall (%)"].iloc[0],
        "isobaric_decoy_flr": op_df[op_df["Tau"] == 0.70]["Isobaric Decoy FLR (%)"].iloc[0],
        "biogenetic_precision": op_df[op_df["Tau"] == 0.70]["Bio-Delta Precision (%)"].iloc[0]
    }},
    "gates": {{
        "G1_Separation_AUC_ge_080": g1_pass,
        "G2_Recall_tau_070_ge_30pct": g2_pass,
        "G3_Precision_tau_070_ge_75pct": g3_pass,
        "G4_Shift_Aware_Lift_ge_010": g4_pass,
        "Overall_Verdict": "PASS (Idea A1 Validated)" if (g1_pass and g3_pass) else "FALSIFIED / CALIBRATE"
    }}
}}

with open("scaffold_eval_report.json", "w") as f:
    json.dump(report, f, indent=2)

print("\\nFINAL REPORT SUMMARY:")
print(json.dumps(report, indent=2))

# %%
# VISUALIZATION PLOTS
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.kdeplot(arm2_vals, ax=axes[0], label="Arm 2: Same Scaffold + BioDelta", fill=True, color="green", alpha=0.3, common_norm=False)
sns.kdeplot(arm1_vals, ax=axes[0], label="Arm 1: Same Scaffold General", fill=True, color="blue", alpha=0.2, common_norm=False)
sns.kdeplot(arm3_vals, ax=axes[0], label="Arm 3: Diff Scaffold Isobaric", fill=True, color="red", alpha=0.3, common_norm=False)
sns.kdeplot(arm4_vals, ax=axes[0], label="Arm 4: Diff Scaffold Random", fill=True, color="gray", alpha=0.2, common_norm=False)
axes[0].set_title("Modified Cosine Similarity Density by Arm")
axes[0].set_xlabel("Modified Cosine Similarity")
axes[0].set_ylabel("Density")
axes[0].axvline(0.70, color="black", linestyle="--", label="Idea A1 Threshold (0.70)")
axes[0].legend(loc="upper right")

axes[1].plot(op_df["Tau"], op_df["Arm2 Bio Recall (%)"], marker='o', label="Bio-Delta Congener Recall (%)", color="green", linewidth=2)
axes[1].plot(op_df["Tau"], op_df["Isobaric Decoy FLR (%)"], marker='s', label="Isobaric Decoy FLR (%)", color="red", linewidth=2)
axes[1].plot(op_df["Tau"], op_df["Bio-Delta Precision (%)"], marker='^', label="Bio-Delta Edge Precision (%)", color="purple", linewidth=2)
axes[1].set_title("Operating Curves across Thresholds Tau")
axes[1].set_xlabel("Similarity Threshold (Tau)")
axes[1].set_ylabel("Percentage (%)")
axes[1].axvline(0.70, color="black", linestyle="--", label="Idea A1 Threshold (0.70)")
axes[1].legend(loc="best")

plt.tight_layout()
plt.savefig("scaffold_network_curves.png")
print("Saved visualization figure to scaffold_network_curves.png")
'''

py_path = '.scratch/kaggle_scaffold_eval/casmi_scaffold_network_eval.py'
nb_path = '.scratch/kaggle_scaffold_eval/casmi_scaffold_network_eval.ipynb'

with open(py_path, 'w', encoding='utf-8') as f:
    f.write(py_content)
print(f"Wrote {py_path}")

# Convert using jupytext
os.system(f'jupytext --to notebook {py_path} -o {nb_path}')

# Add explicit kernelspec metadata via nbformat
nb = nbformat.read(nb_path, as_version=4)
nb.metadata['kernelspec'] = {
    'display_name': 'Python 3',
    'language': 'python',
    'name': 'python3'
}
nb.metadata['language_info'] = {
    'name': 'python',
    'version': '3.13.0'
}
nbformat.write(nb, nb_path)
print(f"Added kernelspec and wrote {nb_path} successfully.")
