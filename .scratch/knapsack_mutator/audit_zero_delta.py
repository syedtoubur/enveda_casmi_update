import pickle, collections, pandas as pd
from rdkit import Chem, RDLogger; RDLogger.DisableLog("rdApp.*")
from rdkit.Chem.Descriptors import ExactMolWt
p=pd.read_parquet("E:/casmi/c3cal_check/pseudo_c3.parquet"); T=dict(zip(p.molecule_id,p.inchikey14)); F=dict(zip(p.molecule_id,p.formula))
R=pickle.load(open(".scratch/knapsack_mutator/full_batch_results.pkl","rb"))
s=pickle.load(open("research/harness/dumps/pseudo_c3/post_dump/state.pkl","rb"))
print("base list lengths", collections.Counter(len(v[1]) for v in s['base'].values()))
print("state keys detail:", {k:type(v).__name__ for k,v in s.items()})
br=R['base_ranks']; mf=R['metfrag_ranks']
gain=[m for m in R['mids'] if (br[m] is None or br[m]>25) and mf[m] and mf[m]<=25]
for m in gain:
    hit=[mu for mu in R['mol_mutants_dict'][m] if mu['inchikey14']==T[m]][0]
    sm=ExactMolWt(Chem.MolFromSmiles(hit['seed_smiles']))
    print(m, F[m], "combo", hit['combo'], "seed-target dM=%.4f"%(hit['mass']-sm), "nmut", len(R['mol_mutants_dict'][m]), "mfrank", mf[m])
# how many mutants share target formula on avg -> isomer pool size
