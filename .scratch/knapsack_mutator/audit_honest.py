import pickle, random, pandas as pd, numpy as np
p=pd.read_parquet("E:/casmi/c3cal_check/pseudo_c3.parquet"); T=dict(zip(p.molecule_id,p.inchikey14))
R=pickle.load(open(".scratch/knapsack_mutator/full_batch_results.pkl","rb"))
base=pickle.load(open("research/harness/dumps/pseudo_c3/post_dump/state.pkl","rb"))['base']
mids=R['mids']; N=len(mids)
def rr(lst,t):
    l=lst[:25]; return 1/(l.index(t)+1) if t in l else 0
def dedup(xs):
    s=set(); o=[]
    for x in xs:
        if x not in s: s.add(x); o.append(x)
    return o
b=[rr(list(base[m][1]),T[m]) for m in mids]
print("base MRR %.4f  top25 %d"%(np.mean(b), sum(x>0 for x in b)))
print("reported oracle union", R['stats']['union_mrr'])
short=sum(len(base[m][1])<25 for m in mids); print("base lists <25:", short)
for name,order in [("gen-order",lambda L:L),("random",None)]:
    vals=[]
    reps = 1 if order else 50
    for rep in range(reps):
        rng=random.Random(rep); tot=[]
        for m in mids:
            ks=[mu['inchikey14'] for mu in R['mol_mutants_dict'][m]]
            if order is None: rng.shuffle(ks)
            tot.append(rr(dedup(list(base[m][1])[:25]+ks),T[m]))
        vals.append(np.mean(tot))
    print("backfill (base then knapsack %s): MRR %.4f"%(name,np.mean(vals)))
# knapsack-first-into-slot k (insert at slot 2, like c3-gen-slot2)
for k in (1,2,5,10):
    tot=[]
    for m in mids:
        ks=[mu['inchikey14'] for mu in R['mol_mutants_dict'][m]]
        bl=list(base[m][1])
        tot.append(rr(dedup(bl[:k-1]+ks[:1]+bl[k-1:]+ks[1:]),T[m]))
    print("insert knapsack#1(gen order) at slot",k,"MRR %.4f"%np.mean(tot))
# how many backfill gains: target in knapsack but not base
g=[m for m in mids if T[m] not in list(base[m][1])[:25] and T[m] in [mu['inchikey14'] for mu in R['mol_mutants_dict'][m]]]
print("gains possible (target in knapsack, not in base top25):",len(g), "of which base list <25:", sum(len(base[m][1])<25 for m in g))
