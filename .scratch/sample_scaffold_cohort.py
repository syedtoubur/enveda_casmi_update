import pandas as pd
import numpy as np
import json
import time

print("Loading train_meta.parquet...")
t0 = time.time()
meta = pd.read_parquet('.scratch/valgen/train_meta.parquet')
print(f"Loaded {len(meta):,} rows in {time.time()-t0:.1f}s.")

# Filter for clean [M+H]+ spectra with >= 6 peaks and valid natural/drug-like scaffolds
# Exclude empty (acyclic) and bare benzene
clean = meta[
    (meta['adduct'] == '[M+H]+') & 
    (meta['n_peaks'] >= 6) & 
    (~meta['scaffold'].isin(['', 'c1ccccc1']))
].copy()

print(f"Clean candidate pool: {len(clean):,} spectra across {clean['scaffold'].nunique():,} scaffolds and {clean['ik14'].nunique():,} molecules")

# Scaffolds with at least 2 distinct molecules
scaf_mol_counts = clean.groupby('scaffold')['ik14'].nunique()
multi_scafs = set(scaf_mol_counts[scaf_mol_counts >= 2].index)
clean_multi = clean[clean['scaffold'].isin(multi_scafs)].copy()

print(f"Multi-molecule pool: {len(clean_multi):,} spectra across {len(multi_scafs):,} scaffolds")

# 17 Biogenetic Mass Shifts (in Da)
BIO_SHIFTS = {
    'CH2': 14.0157,
    'O': 15.9949,
    'H2O': 18.0106,
    'CO': 27.9949,
    'OCH2': 30.0106,
    'C2H2O': 42.0106,  # acetyl
    'CO2': 43.9898,
    'C3H4O2': 72.0211,
    'hexose': 162.0528,
    'rhamnose': 146.0579,
    'glucuronide': 176.0321,
    'pentose': 132.0423,
    'coumaroyl': 146.0368,
    'sinapoyl': 206.0579,
    'feruloyl': 176.0473,
    'caffeoyl': 162.0317,
    'galloyl': 152.0110
}
bio_mass_vals = np.array(list(BIO_SHIFTS.values()), dtype=np.float64)

# Pre-index by precursor_mz for fast isobaric searching
all_pmz = clean['precursor_mz'].to_numpy(dtype=np.float64)
all_rows = clean['row'].to_numpy(dtype=np.int64)
all_scafs = clean['scaffold'].to_numpy()
all_iks = clean['ik14'].to_numpy()

order = np.argsort(all_pmz)
all_pmz_sorted = all_pmz[order]
all_rows_sorted = all_rows[order]
all_scafs_sorted = all_scafs[order]
all_iks_sorted = all_iks[order]

print("Assembling balanced evaluation pairs...")
rng = np.random.default_rng(20261015)

# Group rows by scaffold
scaf_to_indices = clean_multi.groupby('scaffold').indices
scaf_keys = list(scaf_to_indices.keys())
rng.shuffle(scaf_keys)

arm1_pairs = [] # Same scaffold, diff ik14
arm2_pairs = [] # Same scaffold + biogenetic mass defect
arm3_pairs = [] # Diff scaffold, isobaric decoy (<= 20 ppm)
arm4_pairs = [] # Diff scaffold, random decoy

TARGET_SAME = 3000
TARGET_BIO = 1500
TARGET_DEC = 3000

for scaf in scaf_keys:
    idx = scaf_to_indices[scaf]
    sub = clean_multi.iloc[idx]
    
    # Get distinct molecules in this scaffold
    mols = sub.groupby('ik14')
    if len(mols) < 2:
        continue
    
    mol_keys = list(mols.groups.keys())
    # Sample pairs of different molecules with same scaffold
    for _ in range(min(5, len(mol_keys))):
        if len(arm1_pairs) >= TARGET_SAME and len(arm2_pairs) >= TARGET_BIO:
            break
        m1, m2 = rng.choice(mol_keys, size=2, replace=False)
        r1 = rng.choice(mols.groups[m1])
        r2 = rng.choice(mols.groups[m2])
        
        row1 = int(clean_multi.loc[r1, 'row'])
        row2 = int(clean_multi.loc[r2, 'row'])
        pmz1 = float(clean_multi.loc[r1, 'precursor_mz'])
        pmz2 = float(clean_multi.loc[r2, 'precursor_mz'])
        delta_m = abs(pmz1 - pmz2)
        
        # Check if delta_m matches a biogenetic transform within +- 5 ppm
        # (relative to precursor mass ~ 350 Da -> 5 ppm ~ 0.0017 Da)
        min_err = np.min(np.abs(delta_m - bio_mass_vals))
        is_bio = min_err <= 0.015
        
        if is_bio and len(arm2_pairs) < TARGET_BIO:
            arm2_pairs.append({
                "row_A": row1,
                "row_B": row2,
                "pmz_A": pmz1,
                "pmz_B": pmz2,
                "delta_m": delta_m,
                "scaffold": scaf,
                "arm": "Arm2_SameScaffold_BioDelta"
            })
        elif len(arm1_pairs) < TARGET_SAME:
            arm1_pairs.append({
                "row_A": row1,
                "row_B": row2,
                "pmz_A": pmz1,
                "pmz_B": pmz2,
                "delta_m": delta_m,
                "scaffold": scaf,
                "arm": "Arm1_SameScaffold_General"
            })

    if len(arm1_pairs) >= TARGET_SAME and len(arm2_pairs) >= TARGET_BIO:
        break

print(f"Sampled {len(arm1_pairs)} Arm 1 pairs (Same Scaffold General).")
print(f"Sampled {len(arm2_pairs)} Arm 2 pairs (Same Scaffold Bio-Delta).")

# Now sample Arm 3 (Isobaric Decoys <= 20 ppm) and Arm 4 (Random Decoys)
# Use queries from Arm 1 & 2 as anchors
anchor_rows = [p['row_A'] for p in arm1_pairs[:TARGET_DEC]]

for r_anc in anchor_rows:
    # Look up anchor in clean
    anc_row_data = clean[clean['row'] == r_anc].iloc[0]
    pmz = float(anc_row_data['precursor_mz'])
    scaf_anc = anc_row_data['scaffold']
    ik_anc = anc_row_data['ik14']
    
    # Isobaric search window: +- 20 ppm
    tol = pmz * 20e-6
    left = np.searchsorted(all_pmz_sorted, pmz - tol, side='left')
    right = np.searchsorted(all_pmz_sorted, pmz + tol, side='right')
    
    cands_row = all_rows_sorted[left:right]
    cands_scaf = all_scafs_sorted[left:right]
    cands_ik = all_iks_sorted[left:right]
    cands_pmz = all_pmz_sorted[left:right]
    
    # Exclude same scaffold and same ik14
    diff_mask = (cands_scaf != scaf_anc) & (cands_ik != ik_anc)
    if np.any(diff_mask) and len(arm3_pairs) < TARGET_DEC:
        valid_cands_row = cands_row[diff_mask]
        valid_cands_pmz = cands_pmz[diff_mask]
        pick_idx = rng.integers(len(valid_cands_row))
        arm3_pairs.append({
            "row_A": int(r_anc),
            "row_B": int(valid_cands_row[pick_idx]),
            "pmz_A": pmz,
            "pmz_B": float(valid_cands_pmz[pick_idx]),
            "delta_m": abs(pmz - float(valid_cands_pmz[pick_idx])),
            "scaffold_A": scaf_anc,
            "scaffold_B": str(cands_scaf[diff_mask][pick_idx]),
            "arm": "Arm3_DiffScaffold_IsobaricDecoy"
        })
        
    # Sample random decoy (Arm 4)
    if len(arm4_pairs) < TARGET_DEC:
        rand_idx = rng.integers(len(clean))
        rand_row = clean.iloc[rand_idx]
        if rand_row['scaffold'] != scaf_anc:
            arm4_pairs.append({
                "row_A": int(r_anc),
                "row_B": int(rand_row['row']),
                "pmz_A": pmz,
                "pmz_B": float(rand_row['precursor_mz']),
                "delta_m": abs(pmz - float(rand_row['precursor_mz'])),
                "scaffold_A": scaf_anc,
                "scaffold_B": str(rand_row['scaffold']),
                "arm": "Arm4_DiffScaffold_RandomDecoy"
            })

print(f"Sampled {len(arm3_pairs)} Arm 3 pairs (Isobaric Decoys <= 20 ppm).")
print(f"Sampled {len(arm4_pairs)} Arm 4 pairs (Random Decoys).")

all_pairs = arm1_pairs + arm2_pairs + arm3_pairs + arm4_pairs
unique_rows = sorted(set([p['row_A'] for p in all_pairs] + [p['row_B'] for p in all_pairs]))
print(f"Total pairs assembled: {len(all_pairs):,}")
print(f"Total unique spectra rows to fetch from train.parquet: {len(unique_rows):,}")

# Save to json
out_data = {
    "num_pairs": len(all_pairs),
    "unique_rows": unique_rows,
    "pairs": all_pairs
}

with open(".scratch/cohort_scaffold_pairs.json", "w") as f:
    json.dump(out_data, f)
print("Saved .scratch/cohort_scaffold_pairs.json successfully.")
