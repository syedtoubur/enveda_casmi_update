import json
import pandas as pd
import numpy as np
from collections import Counter, defaultdict

# Load data
df = pd.read_csv(r'specs/track4_priors_probes/c3_cal_v2_results/c3_cal_rows.csv')
with open(r'specs/track4_priors_probes/c3_cal_v2_results/mut_lists.json', 'r', encoding='utf-8') as f:
    mut_data = json.load(f)
with open(r'specs/track4_priors_probes/c3_cal_v2_results/pc_lists.json', 'r', encoding='utf-8') as f:
    pc_data = json.load(f)

print("="*80)
print("1. CANDIDATE YIELD & SANITY CHECKS")
print("="*80)
n_cands = [len(v['smiles']) for v in mut_data.values()]
print(f"Total queries in mut_lists.json: {len(mut_data)}")
print(f"Total generated candidates: {sum(n_cands)}")
print(f"Candidates per query: min={min(n_cands)}, max={max(n_cands)}, mean={np.mean(n_cands):.1f}, median={np.median(n_cands):.1f}")

# Check formula tags and duplicate keys
all_tagged = True
duplicate_keys_found = 0
for qid, v in mut_data.items():
    keys = v['keys']
    if len(keys) != len(set(keys)):
        duplicate_keys_found += 1
    for f in v['formula']:
        if not f.endswith('|gen'):
            all_tagged = False

print(f"All formula strings end with '|gen': {all_tagged}")
print(f"Queries with duplicate InChIKey14s in candidate list: {duplicate_keys_found} (must be 0)")

print("\n" + "="*80)
print("2. RECOVERED TRUTHS BREAKDOWN")
print("="*80)
# Look at df where gen_rank > 0
recovered_df = df[df['gen_rank'] > 0]
print(f"Total recovered truths (gen_rank > 0): {len(recovered_df)}")
print("By cohort:")
print(recovered_df['cohort'].value_counts())
print("\nBy rank:")
print(recovered_df['gen_rank'].value_counts().sort_index())
print("\nBy tier:")
print(recovered_df['tier'].value_counts(dropna=False))

# Look at details of recovered queries
recovered_mols = recovered_df['molecule_id'].tolist()
recovered_details = []
for mid in recovered_mols:
    row = df[df['molecule_id'] == mid].iloc[0]
    item = mut_data.get(mid, {})
    gen_rank = int(row['gen_rank'])
    # rank 1 is index 0
    cand_idx = gen_rank - 1
    if cand_idx < len(item.get('smiles', [])):
        src = item['src'][cand_idx]
        key14 = item['keys'][cand_idx]
        formula = item['formula'][cand_idx]
    else:
        src = "N/A"
        key14 = "N/A"
        formula = "N/A"
    recovered_details.append({
        'mol_id': mid,
        'cohort': row['cohort'],
        'tier': row['tier'],
        'rank': gen_rank,
        'src': src,
        'delta_steps': row['delta_steps'],
        'ana_max': row['ana_max'],
        'lib_exact': row['lib_exact'],
        'fz_truth': row['fz_truth'],
        'fz_best': row['fz_best'],
        'best_pool_fz': row['best_pool_fz'],
        'fz_sep': row['fz_sep']
    })

rec_df_detail = pd.DataFrame(recovered_details)
print(f"\nTransform breakdown for recovered truths (top 10):")
print(rec_df_detail['src'].value_counts().head(10))

print("\n" + "="*80)
print("3. FIRED BUT NOT RECOVERED (PSEUDO-C3 OPPORTUNITY & GAP ANALYSIS)")
print("="*80)
c3_df = df[df['cohort'] == 'c3']
c3_fired = c3_df[c3_df['tier'].isin(['sure', 'likely'])]
print(f"Total c3 fired: {len(c3_fired)}")
c3_fired_rec = c3_fired[c3_fired['gen_rank'] > 0]
c3_fired_not_rec = c3_fired[c3_fired['gen_rank'] == 0]
print(f"c3 fired and recovered: {len(c3_fired_rec)}")
print(f"c3 fired but NOT recovered (gen_rank == 0): {len(c3_fired_not_rec)}")

print("\nDetails of c3 fired but not recovered:")
for idx, r in c3_fired_not_rec.iterrows():
    mid = r['molecule_id']
    item = mut_data.get(mid, {})
    print(f"  {mid}: tier={r['tier']}, fz_truth={r['fz_truth']:.1f}, fz_best={r['fz_best']:.1f}, pool_fz={r['best_pool_fz']:.1f}, delta_steps={r['delta_steps']}, ana_max={r['ana_max']:.3f}, n_gen={len(item.get('smiles',[]))}")

print("\n" + "="*80)
print("4. UNRECOVERED C3 QUERIES (268 QUERIES) - REASONS AND DISTRIBUTION")
print("="*80)
c3_not_rec = c3_df[c3_df['gen_rank'] == 0]
print(f"Total c3 unrecovered: {len(c3_not_rec)}")
print("Gate reasons for c3 unrecovered:")
print(c3_not_rec['gate'].value_counts(dropna=False))

print("\nDistribution of ana_max in c3:")
print("  mean:", c3_df['ana_max'].mean(), "median:", c3_df['ana_max'].median())
print("  ana_max >= 0.50 count:", (c3_df['ana_max'] >= 0.50).sum(), "out of 300")
print("  ana_max >= 0.40 count:", (c3_df['ana_max'] >= 0.40).sum(), "out of 300")
print("  ana_max >= 0.30 count:", (c3_df['ana_max'] >= 0.30).sum(), "out of 300")

print("\nDistribution of delta_steps in c3:")
print(c3_df['delta_steps'].value_counts(dropna=False))

print("\n" + "="*80)
print("5. PUBCHEM PROBE (PC_LISTS.JSON) ANALYSIS")
print("="*80)
pc_lens = [len(v.get('pc', [])) for v in pc_data.values()]
print(f"Total entries in pc_data: {len(pc_data)}")
print(f"pc candidate count per query: 0 count={(np.array(pc_lens)==0).sum()}, >0 count={(np.array(pc_lens)>0).sum()}")
print(f"Non-zero pc lens: min={min([l for l in pc_lens if l>0])}, max={max(pc_lens)}, mean={np.mean([l for l in pc_lens if l>0]):.1f}")

# Cohort breakdown in pc_data
pc_lens_by_cohort = defaultdict(list)
for mid, v in pc_data.items():
    row = df[df['molecule_id'] == mid]
    if len(row) > 0:
        c = row.iloc[0]['cohort']
        pc_lens_by_cohort[c].append(len(v.get('pc', [])))

print("Non-zero pc candidates count by cohort:")
for c, lens in pc_lens_by_cohort.items():
    non_zero = sum(1 for l in lens if l > 0)
    print(f"  {c}: {non_zero}/{len(lens)} queries had PubChem candidates within tolerance")
