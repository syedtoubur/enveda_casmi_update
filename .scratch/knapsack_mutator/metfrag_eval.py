"""MetFrag-lite Forward Physics Filter and Fragment Evaluator.

Evaluates in-silico bond-cleavage fragments (max_breaks=2) for generated mutants
against the target molecule and evaluates candidate pruning/ranking efficiency.
"""
from __future__ import annotations
import os
import sys
import time
import pickle
import numpy as np

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath("e:/ENVEDA_casmi"))
from scratch_pkg.casmi_scratch.metfrag import fragment_masses, explain_score

SMOKE_RESULTS_PATH = "e:/ENVEDA_casmi/.scratch/knapsack_mutator/smoke_results.pkl"
EVAL_OUT_PATH = "e:/ENVEDA_casmi/.scratch/knapsack_mutator/metfrag_eval_results.pkl"

def evaluate_metfrag_on_smoke_results():
    print("=" * 65)
    print("METFRAG-LITE FORWARD PHYSICS EVALUATION (SMOKE COHORT)")
    print("=" * 65)
    
    with open(SMOKE_RESULTS_PATH, "rb") as f:
        data = pickle.load(f)
        
    print(f"Loaded {len(data)} queries from smoke test.")
    
    eval_results = {}
    t0 = time.time()
    total_mutants_scored = 0
    rank1_hits = 0
    top5_hits = 0
    top25_hits = 0
    reached_queries = 0
    
    for mid, r in data.items():
        target_smi = r['target_smiles']
        target_ik14 = r['target_ik14']
        mutants = r['mutants']
        is_reached = r['target_reached']
        
        if is_reached:
            reached_queries += 1
            
        if not mutants:
            continue
            
        # Target in-silico fragmentation: acts as synthetic experimental MS2 reference
        target_frags = fragment_masses(target_smi, max_breaks=2)
        if len(target_frags) == 0:
            target_frags = np.array([r['target_mass']])
            
        # Synthetic intensity weighting: simulated daughter ion intensity distribution
        sim_mz = target_frags
        sim_it = np.ones(len(sim_mz), dtype=np.float32)
        
        scored_mutants = []
        for m in mutants:
            smi = m['smiles']
            ik14 = m['inchikey14']
            try:
                frags = fragment_masses(smi, max_breaks=2)
                # Score candidate fragments against the reference spectrum
                score = explain_score(frags, sim_mz, sim_it, mode=1.0, tol=0.01)
            except Exception:
                score = 0.0
                
            scored_mutants.append({
                'smiles': smi,
                'inchikey14': ik14,
                'explain_score': score,
                'steps': m['steps'],
                'is_target': (ik14 == target_ik14),
            })
            total_mutants_scored += 1
            
        # Sort mutants by explain_score descending (physics-based rank)
        scored_mutants.sort(key=lambda x: x['explain_score'], reverse=True)
        
        # Check rank of target
        target_rank = None
        for rank_idx, m in enumerate(scored_mutants):
            if m['is_target']:
                target_rank = rank_idx + 1
                break
                
        eval_results[mid] = {
            'molecule_id': mid,
            'target_ik14': target_ik14,
            'target_reached': is_reached,
            'target_rank': target_rank,
            'num_mutants': len(scored_mutants),
            'top_score': scored_mutants[0]['explain_score'] if scored_mutants else 0.0,
            'target_score': next((m['explain_score'] for m in scored_mutants if m['is_target']), 0.0) if is_reached else None,
        }
        
        if target_rank is not None:
            if target_rank == 1:
                rank1_hits += 1
            if target_rank <= 5:
                top5_hits += 1
            if target_rank <= 25:
                top25_hits += 1
                
        status_str = f"Target Rank #{target_rank}" if target_rank else "Target Not Reached"
        print(f"[{mid}] Mutants: {len(scored_mutants):2d} | {status_str} | Top Score: {eval_results[mid]['top_score']:.3f}")

    elapsed = time.time() - t0
    n = len(data)
    
    print("\n" + "#" * 65)
    print("METFRAG FORWARD PHYSICS BENCHMARK RESULTS")
    print(f"Total evaluated molecules:      {n}")
    print(f"Total mutants scored:           {total_mutants_scored} ({total_mutants_scored/n:.1f} / mol)")
    print(f"Scoring throughput:             {total_mutants_scored/elapsed:.1f} mutants/sec ({elapsed/n:.3f}s / mol)")
    print(f"Ground Truth Reached:           {reached_queries}/{n} ({reached_queries/n*100:.1f}%)")
    print(f"Top-1 Physics Hits:             {rank1_hits}/{n} ({rank1_hits/n*100:.1f}%)")
    print(f"Top-5 Physics Hits:             {top5_hits}/{n} ({top5_hits/n*100:.1f}%)")
    print(f"Top-25 Physics Hits:            {top25_hits}/{n} ({top25_hits/n*100:.1f}%)")
    if reached_queries > 0:
        print(f"Top-1 Precision on Reached:     {rank1_hits}/{reached_queries} ({rank1_hits/reached_queries*100:.1f}%)")
        print(f"Top-25 Precision on Reached:    {top25_hits}/{reached_queries} ({top25_hits/reached_queries*100:.1f}%)")
    print("#" * 65)
    
    with open(EVAL_OUT_PATH, "wb") as f:
        pickle.dump(eval_results, f)
    print(f"\nSaved MetFrag physics evaluation results to: {EVAL_OUT_PATH}")

if __name__ == "__main__":
    evaluate_metfrag_on_smoke_results()
