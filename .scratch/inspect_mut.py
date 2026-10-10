import json
import pandas as pd
from collections import Counter

with open(r'specs/track4_priors_probes/c3_cal_v2_results/mut_lists.json', 'r', encoding='utf-8') as f:
    mut_data = json.load(f)

df = pd.read_csv(r'specs/track4_priors_probes/c3_cal_v2_results/c3_cal_rows.csv')
mol_to_row = df.set_index('molecule_id').to_dict('index')

sure_keys = [k for k, v in mut_data.items() if v.get('tier') == 'sure']
likely_keys = [k for k, v in mut_data.items() if v.get('tier') == 'likely']
empty_keys = [k for k, v in mut_data.items() if not v.get('tier')]

print(f"Sure keys count: {len(sure_keys)}, Likely: {len(likely_keys)}, Empty/Vetoed: {len(empty_keys)}")

def print_summary(k):
    item = mut_data[k]
    row = mol_to_row.get(k, {})
    print(f"\n=== Molecule: {k} (Cohort: {row.get('cohort')}) ===")
    print(f"CSV Row: gate={row.get('gate')}, tier={row.get('tier')}, gen_rank={row.get('gen_rank')}, fz_truth={row.get('fz_truth')}, fz_best={row.get('fz_best')}, best_pool_fz={row.get('best_pool_fz')}")
    print(f"JSON item: tier='{item.get('tier')}'")
    print(f"JSON sig: {json.dumps(item.get('sig', {}), indent=2)}")
    smiles_list = item.get('smiles', [])
    print(f"Number of candidates in JSON: {len(smiles_list)}")
    if smiles_list:
        print("Candidates sample (up to 5):")
        for i in range(min(5, len(smiles_list))):
            smi = item['smiles'][i]
            key = item['keys'][i]
            form = item['formula'][i]
            fz = item['fz'][i]
            src = item['src'][i]
            print(f"  [{i+1}] fz={fz:.2f} | formula={form} | src={src} | key14={key} | smi={smi[:35]}...")

if sure_keys:
    print_summary(sure_keys[0])
if likely_keys:
    print_summary(likely_keys[0])
if empty_keys:
    print_summary(empty_keys[0])
