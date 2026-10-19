"""Extract missed pseudo_c3 molecules (where Engine 1 mutants failed to recover target).

Saves to .scratch/knapsack_mutator/missed_cohort.pkl
"""
import os
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.inchi import MolToInchiKey

P4_PATH = "E:/casmi/c3_gen/p4_inputs.pkl"
PSEUDO_C3_PATH = "E:/casmi/c3cal_check/pseudo_c3.parquet"
OUTPUT_DIR = "e:/ENVEDA_casmi/.scratch/knapsack_mutator"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "missed_cohort.pkl")

def get_ik14(smi):
    try:
        m = Chem.MolFromSmiles(smi)
        if m is not None:
            return MolToInchiKey(m)[:14]
    except Exception:
        pass
    return None

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Loading {P4_PATH}...")
    with open(P4_PATH, "rb") as f:
        p4 = pickle.load(f)
    
    print(f"Loading {PSEUDO_C3_PATH}...")
    p_df = pd.read_parquet(PSEUDO_C3_PATH)
    truth_dict = p_df.set_index("molecule_id").to_dict(orient="index")

    rows = p4["rows"]
    print(f"Processing {len(rows)} molecules from p4_inputs...")

    raw_hits = []
    raw_misses = []
    top25_hits = []
    top25_misses = []
    cohort_records = {}

    for mid, r in rows.items():
        truth = truth_dict.get(mid)
        if not truth:
            print(f"Warning: {mid} not found in ground truth parquet!")
            continue

        target_ik14 = truth["inchikey14"][:14]
        mutants = r.get("mutants", [])

        # Check if target is recovered by Engine 1 mutants (raw mutator reach)
        is_raw_hit = False
        for item in mutants:
            smi = item[0] if isinstance(item, (tuple, list)) else item
            ik14 = get_ik14(smi)
            if ik14 == target_ik14:
                is_raw_hit = True
                break

        # Check top-25 post-ranker reach (first 25 mutants)
        is_top25_hit = False
        for item in mutants[:25]:
            smi = item[0] if isinstance(item, (tuple, list)) else item
            ik14 = get_ik14(smi)
            if ik14 == target_ik14:
                is_top25_hit = True
                break

        if is_raw_hit:
            raw_hits.append(mid)
        else:
            raw_misses.append(mid)

        if is_top25_hit:
            top25_hits.append(mid)
        else:
            top25_misses.append(mid)

        # Record full data for the candidate pipeline
        cohort_records[mid] = {
            "molecule_id": mid,
            "smiles": truth["smiles"],
            "inchikey14": target_ik14,
            "formula": truth["formula"],
            "target_mass": float(r["target"]),
            "fused_top": list(r.get("fused_top", []))[:5],
            "timstof": bool(r.get("timstof", False)),
            "is_raw_hit": is_raw_hit,
            "is_top25_hit": is_top25_hit,
            "adducts": list(truth.get("adducts", [])),
            "ces": list(truth.get("ces", [])),
            "rows": list(truth.get("rows", [])),
        }

    print("\n" + "=" * 55)
    print(f"Total evaluated: {len(rows)}")
    print(f"Raw Mutator Reach:      {len(raw_hits)} hits / {len(raw_misses)} misses ({len(raw_hits)/len(rows)*100:.1f}%)")
    print(f"Post-Ranker Top-25:     {len(top25_hits)} hits / {len(top25_misses)} misses ({len(top25_hits)/len(rows)*100:.1f}%)")
    print("=" * 55)

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump({
            "cohort": cohort_records,
            "raw_misses": raw_misses,
            "raw_hits": raw_hits,
            "top25_misses": top25_misses,
            "top25_hits": top25_hits,
        }, f)

    print(f"\nSuccessfully saved cohort dataset to:")
    print(f"  {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
