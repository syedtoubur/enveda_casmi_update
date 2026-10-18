# The knapsack benchmark (2026-10-16): candidates ranked against the TRUE structure's in-silico fragments.
import pickle
from scratch_pkg.casmi_scratch.metfrag import fragment_masses, explain_score

data = pickle.load(open("results.pkl", "rb"))
for mid, r in data.items():
    target_frags = fragment_masses(r["target_smiles"], max_breaks=2)
    for m in r["mutants"]:
        score = explain_score(fragment_masses(m["smiles"]), target_frags, [1.0] * len(target_frags), mode=1.0)
