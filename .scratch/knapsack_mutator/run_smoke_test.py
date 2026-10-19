"""Run 20-molecule smoke test for Biogenetic Knapsack Mutator.
Evaluates reach, mutant count, and execution time per molecule.
"""
import time
import pickle
import numpy as np
from knapsack_engine import generate_cohort_mutants

COHORT_PATH = "e:/ENVEDA_casmi/.scratch/knapsack_mutator/missed_cohort.pkl"
OUT_PATH = "e:/ENVEDA_casmi/.scratch/knapsack_mutator/smoke_results.pkl"

def main():
    print("=" * 60)
    print("RUNNING 20-MOLECULE SMOKE TEST: BIOGENETIC KNAPSACK MUTATOR")
    print("=" * 60)
    t0 = time.time()
    
    results = generate_cohort_mutants(COHORT_PATH, max_mutants_per_mol=100, max_mols=20)
    
    elapsed = time.time() - t0
    n = len(results)
    per_mol = elapsed / n if n else 0
    
    hits = sum(1 for r in results.values() if r['target_reached'])
    total_mutants = sum(r['num_mutants'] for r in results.values())
    
    print("\n" + "#" * 60)
    print("SMOKE TEST SUMMARY")
    print(f"Total evaluated molecules:     {n}")
    print(f"Total execution time:          {elapsed:.2f}s ({per_mol:.3f}s / molecule)")
    print(f"Total mutants generated:       {total_mutants} (avg {total_mutants/n:.1f} / mol)")
    print(f"Ground Truth Reach:            {hits}/{n} ({hits/n*100:.1f}%)")
    print(f"Budget (<0.200s/mol):          {'PASS' if per_mol < 0.20 else 'MARGINAL'}")
    print("#" * 60)
    
    with open(OUT_PATH, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved smoke test results to: {OUT_PATH}")

if __name__ == "__main__":
    main()
