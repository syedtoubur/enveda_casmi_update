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

# -------------------------------------------------------------------------
# Cell 1: Markdown Title & Experimental Objective
# -------------------------------------------------------------------------
add_md("""# CASMI 2026 Phase 3: Exact Spectral Entropy & Precursor Window-Max Simulation
### Extreme-Value Evaluation of Cross-Instrument Library Matching & The 4-Tier Smart Gate

In Phase 1 and Phase 2, we showed that the standard scalar threshold `lib_max >= 0.90` causes a **97.88% false rejection rate** on cross-instrument timsTOF vs. Orbitrap/Q-TOF pairs, and that multi-dimensional structural filters (precursor masking, $N_{matched} \\ge 4$, high-$m/z$ fragment $> 0.5 \\times M_{prec}$) reduce decoy pairwise false positives by 4.7×.

However, in actual inference, the engine takes the **maximum similarity across all reference spectra within a $\\pm 10$ ppm precursor window**:
$$lib\\_max = \\max_{S \\in \\text{Window}(\\pm 10\\text{ ppm})} \\text{Sim}(Q, S)$$
Because taking the maximum over $K$ candidate decoys is an **extreme value process**, the probability that at least one decoy passes threshold $\\tau$ is:
$$\\mathbb{P}(\\max_{k=1..K} D_k \\ge \\tau) = 1 - \\prod_{k=1}^K (1 - \\text{FPR}_k)$$
If a window contains 20–80 decoys, single-decoy tail events accumulate, inflating the molecule-level false lock rate unless suppressed.

In this Phase 3 experiment, we:
1. Implement exact **Spectral Entropy Similarity** (Li et al., *Nature Methods* 2021) with peak cleaning (`int_floor=0.01`, `max_peaks=64`, $I^{0.5}$, $\\pm 15$ ppm tolerance, precursor masking $\\pm 1.5$ Da):
   $$\\text{Entropy Sim}(P, Q) = 1 - \\frac{2 \\cdot S(M) - S(P) - S(Q)}{\\ln(4)}$$
2. Run realistic **Precursor Window-Max Simulations** across:
   - **Positive Cohort ($N \\approx 1,200$)**: timsTOF queries with a known true match in Domain B reference libraries.
   - **Negative / Decoy Cohort ($N \\approx 1,200$)**: timsTOF queries with zero true matches in Domain B (simulating Class 2 and Class 3 queries).
3. Measure:
   - **Table 1**: Quantiles of True Match Entropy vs. Single Decoy vs. Window-Max Decoy Entropy.
   - **Table 2**: Molecule-Level Operating Curves: Raw Cosine vs. Masked Cosine vs. Spectral Entropy vs. Smart-Filtered Entropy.""")

# -------------------------------------------------------------------------
# Cell 2: Imports and Data Directory Resolution
# -------------------------------------------------------------------------
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

# -------------------------------------------------------------------------
# Cell 3: Metadata Ingestion & Domain Partitioning
# -------------------------------------------------------------------------
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

timstof_keys = set(df_timstof["inchikey14"].dropna())
ref_keys_set = set(df_ref["inchikey14"].dropna())

common_keys = sorted(timstof_keys.intersection(ref_keys_set))
negative_keys = sorted(timstof_keys - ref_keys_set)

print(f"Total timsTOF spectra: {len(df_timstof):,} across {len(timstof_keys):,} structures")
print(f"Total reference spectra: {len(df_ref):,} across {len(ref_keys_set):,} structures")
print(f"Positive overlap structures (Domain A & B): {len(common_keys):,}")
print(f"Negative pure-timsTOF structures (Domain A only, 0 ref match): {len(negative_keys):,}")
""")

# -------------------------------------------------------------------------
# Cell 4: Cohort Sampling & Window Indexing
# -------------------------------------------------------------------------
add_code("""# Index Domain B reference spectra by precursor m/z for fast window queries (+-10 ppm)
ref_mz = df_ref["precursor_mz"].to_numpy(dtype=np.float64)
ref_row = df_ref["row_id"].to_numpy(dtype=np.int64)
ref_keys = df_ref["inchikey14"].to_numpy()
ref_order = np.argsort(ref_mz)
ref_mz_sorted = ref_mz[ref_order]
ref_row_sorted = ref_row[ref_order]
ref_keys_sorted = ref_keys[ref_order]

SAMPLE_SIZE = 1200
rng = np.random.default_rng(20261011)

sampled_pos_keys = rng.choice(common_keys, size=min(SAMPLE_SIZE, len(common_keys)), replace=False)
sampled_neg_keys = rng.choice(negative_keys, size=min(SAMPLE_SIZE, len(negative_keys)), replace=False)

pos_queries = []
neg_queries = []

# Build Positive Cohort
for key in sampled_pos_keys:
    t_group = df_timstof[df_timstof["inchikey14"] == key]
    r_group = df_ref[df_ref["inchikey14"] == key]
    
    # Try to match adduct if available
    common_adducts = set(t_group["adduct"]).intersection(set(r_group["adduct"]))
    if common_adducts:
        ad = list(common_adducts)[0]
        t_row = t_group[t_group["adduct"] == ad].iloc[0]
        r_rows = r_group[r_group["adduct"] == ad]
    else:
        t_row = t_group.iloc[0]
        r_rows = r_group
        
    t_id = int(t_row["row_id"])
    pmz = float(t_row["precursor_mz"])
    
    # Precursor window: +-10 ppm (timsTOF offset centered at +1.45 ppm)
    # Target center is pmz / (1 + 1.45e-6)
    target_center = pmz / (1.0 + 1.45e-6)
    tol = target_center * 10e-6
    left = np.searchsorted(ref_mz_sorted, target_center - tol, side='left')
    right = np.searchsorted(ref_mz_sorted, target_center + tol, side='right')
    
    cand_rows_in_window = ref_row_sorted[left:right]
    cand_keys_in_window = ref_keys_sorted[left:right]
    
    # Partition into true matches and decoys
    true_mask = (cand_keys_in_window == key)
    true_ids = cand_rows_in_window[true_mask].tolist()
    decoy_ids = cand_rows_in_window[~true_mask].tolist()
    
    # If true match happened to be slightly outside +-10 ppm due to measurement noise, add known r_rows
    if len(true_ids) == 0:
        true_ids = [int(r_rows.iloc[0]["row_id"])]
        
    pos_queries.append({
        "query_id": t_id,
        "key": key,
        "pmz": pmz,
        "adduct": t_row["adduct"],
        "true_ids": true_ids,
        "decoy_ids": decoy_ids,
        "n_cand_window": len(cand_rows_in_window),
        "n_decoys": len(decoy_ids)
    })

# Build Negative Cohort (Zero true matches in reference set)
for key in sampled_neg_keys:
    t_row = df_timstof[df_timstof["inchikey14"] == key].iloc[0]
    t_id = int(t_row["row_id"])
    pmz = float(t_row["precursor_mz"])
    
    target_center = pmz / (1.0 + 1.45e-6)
    tol = target_center * 10e-6
    left = np.searchsorted(ref_mz_sorted, target_center - tol, side='left')
    right = np.searchsorted(ref_mz_sorted, target_center + tol, side='right')
    
    decoy_ids = ref_row_sorted[left:right].tolist()
    
    neg_queries.append({
        "query_id": t_id,
        "key": key,
        "pmz": pmz,
        "adduct": t_row["adduct"],
        "decoy_ids": decoy_ids,
        "n_cand_window": len(decoy_ids)
    })

print(f"Positive cohort assembled: {len(pos_queries):,} queries")
pos_cand_counts = [q["n_cand_window"] for q in pos_queries]
print(f"  Positive window candidates: median={np.median(pos_cand_counts):.0f}, mean={np.mean(pos_cand_counts):.1f}, max={np.max(pos_cand_counts)}")

print(f"Negative cohort assembled: {len(neg_queries):,} queries")
neg_cand_counts = [q["n_cand_window"] for q in neg_queries]
print(f"  Negative window candidates: median={np.median(neg_cand_counts):.0f}, mean={np.mean(neg_cand_counts):.1f}, max={np.max(neg_cand_counts)}")
""")

# -------------------------------------------------------------------------
# Cell 5: Selective Parquet Spectral Ingestion
# -------------------------------------------------------------------------
add_code("""# Gather all unique row IDs needed
needed_rows = set()
for q in pos_queries:
    needed_rows.add(q["query_id"])
    needed_rows.update(q["true_ids"])
    needed_rows.update(q["decoy_ids"])
for q in neg_queries:
    needed_rows.add(q["query_id"])
    needed_rows.update(q["decoy_ids"])

all_row_ids = sorted(needed_rows)
print(f"Total unique spectral rows to load from Parquet: {len(all_row_ids):,}")

t0 = time.time()
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

print(f"Loaded {len(row_map):,} spectra into memory in {time.time()-t0:.1f}s.")
""")

# -------------------------------------------------------------------------
# Cell 6: Exact Spectral Entropy Engine & Peak Cleaner
# -------------------------------------------------------------------------
add_code("""def clean_and_prep_spectrum(mzs, ints, prec_mz=None, mask_precursor=True, int_floor=0.01, max_peaks=64, power=0.5):
    \"\"\"
    Cleans raw peaks and computes probability distribution weights and Shannon entropy.
    Parameters:
      - prec_mz: Precursor m/z for precursor masking (|mz - prec_mz| > 1.5 Da).
      - int_floor: Relative threshold (drops peaks < int_floor * max_peak).
      - max_peaks: Keeps top N most intense peaks.
      - power: Intensity exponent (0.5 = square root).
    Returns:
      (cleaned_mzs, weights, entropy, is_empty)
    \"\"\"
    if len(mzs) == 0:
        return np.array([]), np.array([]), 0.0, True
        
    mzs = np.asarray(mzs, dtype=np.float64)
    ints = np.asarray(ints, dtype=np.float64)
    
    if mask_precursor and prec_mz is not None:
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
    
    w = np.power(ints, power)
    w_sum = np.sum(w)
    if w_sum <= 0:
        return np.array([]), np.array([]), 0.0, True
    w = w / w_sum
    
    entropy = -float(np.sum(w * np.log(w + 1e-15)))
    return mzs, w, entropy, False

def compare_spectra(prep1, prep2, prec_mz, mz_tol_ppm=15.0):
    \"\"\"
    Fast comparison between two pre-cleaned spectra.
    Returns:
      entropy_sim: float [0, 1]
      cos_sim: float [0, 1]
      n_matched: int
      has_high_mz: bool
      max_matched_mz: float
    \"\"\"
    m1, w1, s1, empty1 = prep1
    m2, w2, s2, empty2 = prep2
    
    if empty1 or empty2:
        return 0.0, 0.0, 0, False, 0.0
        
    half_prec = 0.5 * prec_mz if prec_mz is not None else 0.0
    j_start = 0
    n2 = len(m2)
    
    matched_j_set = set()
    matches = [] # (i, j, mz1, w1, w2)
    dot = 0.0
    matched_mzs = []
    
    norm1 = np.linalg.norm(w1)
    norm2 = np.linalg.norm(w2)
    
    for i, (mz_val, val1) in enumerate(zip(m1, w1)):
        tol = mz_val * mz_tol_ppm * 1e-6
        best_diff = tol
        best_j = -1
        for j in range(j_start, n2):
            diff = abs(mz_val - m2[j])
            if m2[j] < mz_val - tol:
                j_start = j
                continue
            if m2[j] > mz_val + tol:
                break
            if diff <= best_diff:
                best_diff = diff
                best_j = j
                
        if best_j != -1 and best_j not in matched_j_set:
            matched_j_set.add(best_j)
            val2 = w2[best_j]
            matches.append((i, best_j, mz_val, val1, val2))
            matched_mzs.append(mz_val)
            dot += val1 * val2
            
    n_matched = len(matches)
    has_high_mz = any(m > half_prec for m in matched_mzs)
    max_matched_mz = max(matched_mzs) if matched_mzs else 0.0
    cos_sim = float(dot / (norm1 * norm2)) if (norm1 > 0 and norm2 > 0) else 0.0
    
    # Mixture Spectrum M = (P + Q) / 2
    matched_i_set = {m[0] for m in matches}
    mixture_weights = []
    
    for _, _, _, v1, v2 in matches:
        mixture_weights.append((v1 + v2) * 0.5)
        
    for i, v1 in enumerate(w1):
        if i not in matched_i_set:
            mixture_weights.append(v1 * 0.5)
            
    for j, v2 in enumerate(w2):
        if j not in matched_j_set:
            mixture_weights.append(v2 * 0.5)
            
    mixture_weights = np.asarray(mixture_weights, dtype=np.float64)
    s_mix = -float(np.sum(mixture_weights * np.log(mixture_weights + 1e-15)))
    
    # Entropy distance D = 2 * S(M) - S(P) - S(Q)
    entropy_dist = max(0.0, 2.0 * s_mix - s1 - s2)
    entropy_sim = float(max(0.0, min(1.0, 1.0 - (entropy_dist / np.log(4.0)))))
    
    return entropy_sim, cos_sim, n_matched, has_high_mz, max_matched_mz

print("Spectral Entropy matching engine compiled and ready.")
""")

# -------------------------------------------------------------------------
# Cell 7: Window-Max Simulation Execution
# -------------------------------------------------------------------------
add_code("""print("Executing Precursor Window-Max Simulation across Cohorts...")
t0 = time.time()

# 1. Evaluate Positive Cohort
pos_results = []

for q in pos_queries:
    pmz = q["pmz"]
    raw_qmz, raw_qit = row_map[q["query_id"]]
    
    # Pre-clean query spectra (both masked and unmasked)
    q_masked = clean_and_prep_spectrum(raw_qmz, raw_qit, prec_mz=pmz, mask_precursor=True)
    q_raw = clean_and_prep_spectrum(raw_qmz, raw_qit, prec_mz=pmz, mask_precursor=False)
    
    # Evaluate True Matches in window
    best_true = {
        "entropy_masked": 0.0, "cos_masked": 0.0,
        "entropy_raw": 0.0, "cos_raw": 0.0,
        "n_matched": 0, "has_high_mz": False
    }
    for tid in q["true_ids"]:
        if tid not in row_map: continue
        tmz, tit = row_map[tid]
        t_masked = clean_and_prep_spectrum(tmz, tit, prec_mz=pmz, mask_precursor=True)
        t_raw = clean_and_prep_spectrum(tmz, tit, prec_mz=pmz, mask_precursor=False)
        
        ent_m, cos_m, nm_m, high_m, _ = compare_spectra(q_masked, t_masked, pmz)
        ent_r, cos_r, nm_r, high_r, _ = compare_spectra(q_raw, t_raw, pmz)
        
        if ent_m > best_true["entropy_masked"]:
            best_true["entropy_masked"] = ent_m
            best_true["cos_masked"] = cos_m
            best_true["entropy_raw"] = ent_r
            best_true["cos_raw"] = cos_r
            best_true["n_matched"] = nm_m
            best_true["has_high_mz"] = high_m
            
    # Evaluate Decoys in window
    decoy_entropy_raw_max = 0.0
    decoy_cos_raw_max = 0.0
    decoy_entropy_masked_max = 0.0
    decoy_cos_masked_max = 0.0
    
    # Decoys passing Smart Filter: N_matched >= 4 and has_high_mz
    smart_decoy_entropy_max = 0.0
    smart_decoy_cos_max = 0.0
    
    # Single random decoy comparison
    single_decoy_ent = 0.0
    single_decoy_cos = 0.0
    
    if q["decoy_ids"]:
        # Pick first decoy as single-decoy baseline
        d0_id = q["decoy_ids"][0]
        if d0_id in row_map:
            dmz, dit = row_map[d0_id]
            d_masked = clean_and_prep_spectrum(dmz, dit, prec_mz=pmz, mask_precursor=True)
            ent_0, cos_0, _, _, _ = compare_spectra(q_masked, d_masked, pmz)
            single_decoy_ent = ent_0
            single_decoy_cos = cos_0
            
        for did in q["decoy_ids"]:
            if did not in row_map: continue
            dmz, dit = row_map[did]
            d_masked = clean_and_prep_spectrum(dmz, dit, prec_mz=pmz, mask_precursor=True)
            d_raw = clean_and_prep_spectrum(dmz, dit, prec_mz=pmz, mask_precursor=False)
            
            ent_m, cos_m, nm_m, high_m, _ = compare_spectra(q_masked, d_masked, pmz)
            ent_r, cos_r, _, _, _ = compare_spectra(q_raw, d_raw, pmz)
            
            if ent_r > decoy_entropy_raw_max: decoy_entropy_raw_max = ent_r
            if cos_r > decoy_cos_raw_max: decoy_cos_raw_max = cos_r
            if ent_m > decoy_entropy_masked_max: decoy_entropy_masked_max = ent_m
            if cos_m > decoy_cos_masked_max: decoy_cos_masked_max = cos_m
            
            # Smart filter condition: N_matched >= 4 and has_high_mz
            if nm_m >= 4 and high_m:
                if ent_m > smart_decoy_entropy_max: smart_decoy_entropy_max = ent_m
                if cos_m > smart_decoy_cos_max: smart_decoy_cos_max = cos_m

    # True match beats window decoys?
    true_beats_entropy = (best_true["entropy_masked"] > decoy_entropy_masked_max)
    true_beats_cos = (best_true["cos_masked"] > decoy_cos_masked_max)

    pos_results.append({
        "query_id": q["query_id"],
        "key": q["key"],
        "pmz": pmz,
        "n_cand_window": q["n_cand_window"],
        "n_decoys": q["n_decoys"],
        # True match metrics
        "true_entropy_masked": best_true["entropy_masked"],
        "true_cos_masked": best_true["cos_masked"],
        "true_entropy_raw": best_true["entropy_raw"],
        "true_cos_raw": best_true["cos_raw"],
        "true_n_matched": best_true["n_matched"],
        "true_has_high_mz": best_true["has_high_mz"],
        # Single decoy baseline
        "single_decoy_entropy": single_decoy_ent,
        "single_decoy_cos": single_decoy_cos,
        # Window-max decoy metrics
        "decoy_entropy_raw_max": decoy_entropy_raw_max,
        "decoy_cos_raw_max": decoy_cos_raw_max,
        "decoy_entropy_masked_max": decoy_entropy_masked_max,
        "decoy_cos_masked_max": decoy_cos_masked_max,
        "smart_decoy_entropy_max": smart_decoy_entropy_max,
        "smart_decoy_cos_max": smart_decoy_cos_max,
        # Win flags
        "true_beats_entropy": true_beats_entropy,
        "true_beats_cos": true_beats_cos
    })

# 2. Evaluate Negative Cohort (0 true matches; purely isobaric decoys)
neg_results = []

for q in neg_queries:
    pmz = q["pmz"]
    raw_qmz, raw_qit = row_map[q["query_id"]]
    
    q_masked = clean_and_prep_spectrum(raw_qmz, raw_qit, prec_mz=pmz, mask_precursor=True)
    q_raw = clean_and_prep_spectrum(raw_qmz, raw_qit, prec_mz=pmz, mask_precursor=False)
    
    decoy_entropy_raw_max = 0.0
    decoy_cos_raw_max = 0.0
    decoy_entropy_masked_max = 0.0
    decoy_cos_masked_max = 0.0
    smart_decoy_entropy_max = 0.0
    smart_decoy_cos_max = 0.0
    single_decoy_ent = 0.0
    single_decoy_cos = 0.0
    
    if q["decoy_ids"]:
        d0_id = q["decoy_ids"][0]
        if d0_id in row_map:
            dmz, dit = row_map[d0_id]
            d_masked = clean_and_prep_spectrum(dmz, dit, prec_mz=pmz, mask_precursor=True)
            ent_0, cos_0, _, _, _ = compare_spectra(q_masked, d_masked, pmz)
            single_decoy_ent = ent_0
            single_decoy_cos = cos_0
            
        for did in q["decoy_ids"]:
            if did not in row_map: continue
            dmz, dit = row_map[did]
            d_masked = clean_and_prep_spectrum(dmz, dit, prec_mz=pmz, mask_precursor=True)
            d_raw = clean_and_prep_spectrum(dmz, dit, prec_mz=pmz, mask_precursor=False)
            
            ent_m, cos_m, nm_m, high_m, _ = compare_spectra(q_masked, d_masked, pmz)
            ent_r, cos_r, _, _, _ = compare_spectra(q_raw, d_raw, pmz)
            
            if ent_r > decoy_entropy_raw_max: decoy_entropy_raw_max = ent_r
            if cos_r > decoy_cos_raw_max: decoy_cos_raw_max = cos_r
            if ent_m > decoy_entropy_masked_max: decoy_entropy_masked_max = ent_m
            if cos_m > decoy_cos_masked_max: decoy_cos_masked_max = cos_m
            
            if nm_m >= 4 and high_m:
                if ent_m > smart_decoy_entropy_max: smart_decoy_entropy_max = ent_m
                if cos_m > smart_decoy_cos_max: smart_decoy_cos_max = cos_m

    neg_results.append({
        "query_id": q["query_id"],
        "key": q["key"],
        "pmz": pmz,
        "n_cand_window": q["n_cand_window"],
        "single_decoy_entropy": single_decoy_ent,
        "single_decoy_cos": single_decoy_cos,
        "decoy_entropy_raw_max": decoy_entropy_raw_max,
        "decoy_cos_raw_max": decoy_cos_raw_max,
        "decoy_entropy_masked_max": decoy_entropy_masked_max,
        "decoy_cos_masked_max": decoy_cos_masked_max,
        "smart_decoy_entropy_max": smart_decoy_entropy_max,
        "smart_decoy_cos_max": smart_decoy_cos_max
    })

df_pos = pd.DataFrame(pos_results)
df_neg = pd.DataFrame(neg_results)
print(f"Simulation completed in {time.time()-t0:.1f}s.")
print(f"Evaluated {len(df_pos):,} Positive queries and {len(df_neg):,} Negative queries.")
""")

# -------------------------------------------------------------------------
# Cell 8: Table 1 - Quantile Distributions (Quantifying Extreme-Value Inflation)
# -------------------------------------------------------------------------
add_code("""print("\\n" + "="*85)
print("TABLE 1: QUANTILE DISTRIBUTION ANALYSIS (TRUE MATCH VS. DECOY WINDOW-MAX)")
print("="*85)

quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

series_dict = {
    "1. True Match Raw Cosine": df_pos["true_cos_raw"],
    "2. True Match Masked Cosine": df_pos["true_cos_masked"],
    "3. True Match Spectral Entropy": df_pos["true_entropy_masked"],
    "4. Single Decoy Masked Cosine": df_pos[df_pos["n_decoys"] > 0]["single_decoy_cos"],
    "5. Single Decoy Spectral Entropy": df_pos[df_pos["n_decoys"] > 0]["single_decoy_entropy"],
    "6. Window-Max Decoy Raw Cosine": df_neg["decoy_cos_raw_max"],
    "7. Window-Max Decoy Masked Cosine": df_neg["decoy_cos_masked_max"],
    "8. Window-Max Decoy Spectral Entropy": df_neg["decoy_entropy_masked_max"],
    "9. Window-Max Decoy Smart-Filtered Entropy": df_neg["smart_decoy_entropy_max"]
}

q_rows = []
for name, s in series_dict.items():
    vals = np.quantile(s, quantiles)
    row = {"Metric Series": name}
    for q, v in zip(quantiles, vals):
        row[f"p{int(q*100):02d}"] = round(float(v), 4)
    row["Mean"] = round(float(np.mean(s)), 4)
    row["Max"] = round(float(np.max(s)), 4)
    q_rows.append(row)

df_quantiles = pd.DataFrame(q_rows)
print(df_quantiles.to_string(index=False))

os.makedirs("/kaggle/working", exist_ok=True)
df_quantiles.to_csv("/kaggle/working/phase3_quantile_distribution.csv", index=False)
""")

# -------------------------------------------------------------------------
# Cell 9: Table 2 - Molecule-Level Operating Curves across Thresholds
# -------------------------------------------------------------------------
add_code("""print("\\n" + "="*95)
print("TABLE 2: MOLECULE-LEVEL OPERATING CURVES (TRUE MATCH RECALL VS. WINDOW-MAX FALSE LOCK RATE)")
print("="*95)

thresholds = [0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40]
op_results = []

for tau in thresholds:
    # True Match Recall: Did the true match meet threshold tau?
    recall_raw_cos = (df_pos["true_cos_raw"] >= tau).mean() * 100.0
    recall_masked_cos = (df_pos["true_cos_masked"] >= tau).mean() * 100.0
    recall_entropy = (df_pos["true_entropy_masked"] >= tau).mean() * 100.0
    recall_smart = ((df_pos["true_entropy_masked"] >= tau) & 
                    (df_pos["true_n_matched"] >= 4) & 
                    (df_pos["true_has_high_mz"])).mean() * 100.0
    
    # Molecule-Level False Lock Rate on Negative Cohort (queries with 0 true matches)
    # A False Lock occurs if ANY candidate in the window passes threshold tau
    flr_raw_cos = (df_neg["decoy_cos_raw_max"] >= tau).mean() * 100.0
    flr_masked_cos = (df_neg["decoy_cos_masked_max"] >= tau).mean() * 100.0
    flr_entropy = (df_neg["decoy_entropy_masked_max"] >= tau).mean() * 100.0
    flr_smart = (df_neg["smart_decoy_entropy_max"] >= tau).mean() * 100.0
    
    # Single-Decoy FPR for comparison
    single_decoy_fpr = (df_neg["single_decoy_entropy"] >= tau).mean() * 100.0
    
    # Positive Cohort Window Rank 1 Rate:
    # Does true match pass tau AND beat all decoys in window?
    rank1_win_entropy = ((df_pos["true_entropy_masked"] >= tau) & df_pos["true_beats_entropy"]).mean() * 100.0
    rank1_win_cos = ((df_pos["true_cos_masked"] >= tau) & df_pos["true_beats_cos"]).mean() * 100.0
    
    # Precision assuming realistic prior (15% Class 1 library hit rate, 85% Class 2/3 queries)
    # Prec = TP / (TP + FP) = (0.15 * Recall) / (0.15 * Recall + 0.85 * FLR)
    tp_weight = 0.15 * (recall_smart / 100.0)
    fp_weight = 0.85 * (flr_smart / 100.0)
    prec_smart = (tp_weight / (tp_weight + fp_weight + 1e-9)) * 100.0
    
    op_results.append({
        "Threshold (tau)": tau,
        "Raw Cos Recall (%)": round(recall_raw_cos, 2),
        "Raw Cos FLR (%)": round(flr_raw_cos, 2),
        "Masked Cos Recall (%)": round(recall_masked_cos, 2),
        "Masked Cos FLR (%)": round(flr_masked_cos, 2),
        "Entropy Recall (%)": round(recall_entropy, 2),
        "Entropy FLR (%)": round(flr_entropy, 2),
        "Smart Recall (%)": round(recall_smart, 2),
        "Smart FLR (%)": round(flr_smart, 2),
        "Single Decoy FPR (%)": round(single_decoy_fpr, 2),
        "Entropy Rank1 Win (%)": round(rank1_win_entropy, 2),
        "Smart Precision (%)": round(prec_smart, 2)
    })

df_op = pd.DataFrame(op_results)
print(df_op.to_string(index=False))

df_op.to_csv("/kaggle/working/phase3_entropy_operating_curves.csv", index=False)

# Compile structured JSON report
report = {
    "experiment": "Phase 3: Cross-Instrument Spectral Entropy & Precursor Window-Max Simulation",
    "cohort_sizes": {
        "num_positive_queries": len(df_pos),
        "num_negative_queries": len(df_neg),
        "median_pos_window_candidates": float(np.median(df_pos["n_cand_window"])),
        "median_neg_window_candidates": float(np.median(df_neg["n_cand_window"]))
    },
    "key_conclusions": {
        "old_baseline_090_recall": float((df_pos["true_cos_raw"] >= 0.90).mean() * 100.0),
        "old_baseline_090_flr": float((df_neg["decoy_cos_raw_max"] >= 0.90).mean() * 100.0),
        "smart_gate_tau_065_recall": float(((df_pos["true_entropy_masked"] >= 0.65) & (df_pos["true_n_matched"] >= 4) & (df_pos["true_has_high_mz"])).mean() * 100.0),
        "smart_gate_tau_065_flr": float((df_neg["smart_decoy_entropy_max"] >= 0.65).mean() * 100.0),
        "smart_gate_tau_040_recall": float(((df_pos["true_entropy_masked"] >= 0.40) & (df_pos["true_n_matched"] >= 4) & (df_pos["true_has_high_mz"])).mean() * 100.0),
        "smart_gate_tau_040_flr": float((df_neg["smart_decoy_entropy_max"] >= 0.40).mean() * 100.0),
        "extreme_value_inflation_factor_at_050": float((df_neg["decoy_entropy_masked_max"] >= 0.50).mean() / ((df_neg["single_decoy_entropy"] >= 0.50).mean() + 1e-9))
    },
    "quantiles": df_quantiles.to_dict(orient="records"),
    "operating_curves": df_op.to_dict(orient="records")
}

Path("/kaggle/working/phase3_entropy_window_max_report.json").write_text(json.dumps(report, indent=2))
print("\\nSaved report to /kaggle/working/phase3_entropy_window_max_report.json")
""")

# -------------------------------------------------------------------------
# Cell 10: Publication-Grade Visualizations
# -------------------------------------------------------------------------
add_code("""fig, axes = plt.subplots(1, 2, figsize=(16, 6))

def safe_kde(data, ax, color, label, linestyle="-", linewidth=2.0):
    vals = data[np.isfinite(data)]
    if len(vals) > 0 and np.std(vals) > 1e-4:
        sns.kdeplot(vals, ax=ax, color=color, linewidth=linewidth, linestyle=linestyle, label=label)
    else:
        ax.axvline(float(np.median(vals) if len(vals) else 0.0), color=color, linestyle=linestyle, label=label)

# Panel 1: Extreme Value Distribution Comparison
safe_kde(df_pos["true_entropy_masked"], axes[0], "#2a9d8f", "True Match Entropy (Masked)", linewidth=2.5)
safe_kde(df_neg["decoy_entropy_masked_max"], axes[0], "#e76f51", "Window-Max Decoy Entropy (Unfiltered)", linestyle="--", linewidth=2.0)
safe_kde(df_neg["smart_decoy_entropy_max"], axes[0], "#264653", "Window-Max Decoy Smart Gate (N>=4 + High-m/z)", linewidth=2.5)
safe_kde(df_neg[df_neg["n_cand_window"] > 0]["single_decoy_entropy"], axes[0], "#a8dadc", "Single Decoy Entropy (Pairwise)", linestyle=":", linewidth=1.5)

axes[0].axvline(0.90, color="#d62828", linestyle=":", label="Old Baseline (tau = 0.90)")
axes[0].axvline(0.65, color="#f4a261", linestyle="-.", label="Tier 2 Gate (tau = 0.65)")
axes[0].axvline(0.40, color="#2a9d8f", linestyle="-.", label="Tier 3 Gate (tau = 0.40)")

axes[0].set_title("Spectral Entropy Density: True Matches vs. Extreme-Value Window Decoys", fontsize=13)
axes[0].set_xlabel("Spectral Entropy Similarity")
axes[0].set_ylabel("Kernel Density")
axes[0].set_xlim(0.0, 1.05)
axes[0].legend(loc="upper right", fontsize=9)

# Panel 2: Operating Curves (Recall vs. Window-Max False Lock Rate)
axes[1].plot(df_op["Raw Cos FLR (%)"], df_op["Raw Cos Recall (%)"], marker="x", color="#e76f51", linestyle="--", label="Raw Cosine (Baseline)")
axes[1].plot(df_op["Masked Cos FLR (%)"], df_op["Masked Cos Recall (%)"], marker="o", color="#457b9d", linestyle=":", label="Masked Cosine")
axes[1].plot(df_op["Entropy FLR (%)"], df_op["Entropy Recall (%)"], marker="^", color="#f4a261", linestyle="-.", label="Masked Entropy (No Filters)")
axes[1].plot(df_op["Smart FLR (%)"], df_op["Smart Recall (%)"], marker="s", color="#2a9d8f", linewidth=2.5, label="Smart Gate (Entropy + Filters)")

for _, row in df_op.iterrows():
    if row["Threshold (tau)"] in [0.90, 0.65, 0.40]:
        axes[1].annotate(f"tau={row['Threshold (tau)']:.2f}",
                         (row["Smart FLR (%)"], row["Smart Recall (%)"]),
                         textcoords="offset points", xytext=(8, -5), fontsize=9, fontweight="bold", color="#264653")

axes[1].set_title("Operating Curve: Recall vs. Precursor Window-Max False Lock Rate", fontsize=13)
axes[1].set_xlabel("Molecule-Level Window-Max False Lock Rate (%)")
axes[1].set_ylabel("True Match Recall (%)")
axes[1].set_ylim(0, 100)
axes[1].legend(loc="lower right", fontsize=10)

plt.tight_layout()
plt.savefig("/kaggle/working/phase3_entropy_window_max_curves.png", dpi=200)
plt.show()
print("Saved figure to /kaggle/working/phase3_entropy_window_max_curves.png")
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
print(f"Generated Phase 3 Notebook: {target_file} with {len(cells)} cells.")
