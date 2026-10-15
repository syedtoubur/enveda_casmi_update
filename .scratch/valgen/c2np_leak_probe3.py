"""P3: provenance / popularity of the truth inside the BASE pool window (CPU, read-only).
Pool files from ahmedberatozer/casmi26-v2-pool + dmitriigluzdov/casmi26-pubchem-popularity-prior in E:/casmi/dl_pool."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path('E:/ENVEDA_casmi')
D = Path('E:/casmi/dl_pool')
sys.path.insert(0, str(R / 'research/harness'))
import harness as HX  # noqa: E402

meta = pd.read_parquet(D / 'pool_meta.parquet', columns=['mass', 'key', 'src', 'train_sid'])
mass = meta.mass.values.astype(np.float64)
src = meta.src.values
tsid = meta.train_sid.values
pop = np.load(D / 'pool_lsid.npy', allow_pickle=False).astype(np.float64) + np.load(D / 'pool_lpmid.npy', allow_pickle=False).astype(np.float64)
key2pid = dict(zip(meta.key.values, range(len(meta))))
print('pool', len(meta), 'src counts', dict(zip(*np.unique(src, return_counts=True))), 'train_sid>=0', int((tsid >= 0).sum()))
T = HX.truths()
out = {}
for shard in sys.argv[1:] or ['C2_np', 'C2']:
    state, deps, cp = HX.load_dump(shard)
    mids = [m for m in state['te_mids'] if m in T]
    rows = []
    for m in mids:
        smis, keys = state['base'][m][0], state['base'][m][1]
        tk = cp.canonicalize_smiles_to_ik14(T[m]['smiles'])
        top1 = bool(smis) and cp.canonicalize_smiles_to_ik14(smis[0]) == tk
        tpid = next((key2pid.get(k) for k, s in zip(keys, smis) if cp.canonicalize_smiles_to_ik14(s) == tk and k in key2pid), None)
        if tpid is None:
            rows.append(dict(top1=top1, found=False)); continue
        tgt = mass[tpid]
        tol = tgt * 10e-6
        w = np.arange(np.searchsorted(mass, tgt - tol, 'left'), np.searchsorted(mass, tgt + tol, 'right'))
        wp = pop[w]; wtr = tsid[w] >= 0
        blist = np.array([key2pid[k] for k in keys if k in key2pid])
        rows.append(dict(
            top1=top1, found=True, n_win=len(w), n_train_win=int(wtr.sum()), truth_train=bool(tsid[tpid] >= 0),
            truth_src=int(src[tpid]), truth_pop=float(pop[tpid]),
            pop_rank_win=int(1 + (wp > pop[tpid]).sum()),
            pop_rank_list=int(1 + (pop[blist] > pop[tpid]).sum()) if len(blist) else -1,
            only_train=bool(wtr.sum() == 1 and tsid[tpid] >= 0),
            win_pop_med=float(np.median(wp))))
    d = pd.DataFrame(rows); f = d[d.found]
    print(f'\n== {shard}: n {len(d)} truth found in BASE list {d.found.mean():.3f}; BASE top1 {d.top1.mean():.3f}')
    print(f'   truth is train-derived {f.truth_train.mean():.3f}; truth src values {dict(f.truth_src.value_counts())}')
    print(f'   window size median {f.n_win.median():.0f}, train-derived in window median {f.n_train_win.median():.0f}')
    print(f'   truth = ONLY train-derived structure in window: {f.only_train.mean():.3f}')
    print(f'   popularity alone, rank 1 in window: {(f.pop_rank_win == 1).mean():.3f}; rank 1 in BASE list: {(f.pop_rank_list == 1).mean():.3f}')
    print(f'   truth pop median {f.truth_pop.median():.2f} vs window median {f.win_pop_med.median():.2f}')
    print('   BASE top1 | only_train', f.groupby('only_train').top1.agg(['mean', 'size']).to_dict('index'))
    print('   BASE top1 | pop rank1 in window', f.groupby(f.pop_rank_win == 1).top1.agg(['mean', 'size']).to_dict('index'))
    d.to_csv(R / f'research/harness/dumps/{shard}/leak_probe3.csv', index=False)
