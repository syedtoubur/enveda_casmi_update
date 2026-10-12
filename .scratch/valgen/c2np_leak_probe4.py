"""P4: BASE top-1 with the v19 popularity term removed (exact: scs = z + 0.15*POOL_POP[pid] for pool entries,
generated entries keep their slots). Also: top-1 with popularity only, and with MU doubled / halved."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path('C:/Users/h/OneDrive/Desktop/ENVEDA_casmi')
D = Path('C:/casmi/dl_pool')
sys.path.insert(0, str(R / 'research/harness'))
import harness as HX  # noqa: E402

meta = pd.read_parquet(D / 'pool_meta.parquet', columns=['key'])
key2pid = dict(zip(meta.key.values, range(len(meta))))
pop = np.load(D / 'pool_lsid.npy', allow_pickle=False).astype(np.float64) + np.load(D / 'pool_lpmid.npy', allow_pickle=False).astype(np.float64)
T = HX.truths()
for shard in sys.argv[1:] or ['C2_np', 'C2', 'C1_npx']:
    state, deps, cp = HX.load_dump(shard)
    mids = [m for m in state['te_mids'] if m in T]
    res = {mu: [] for mu in ('as_run', 0.0, 0.075, 0.30, 'pop_only')}
    for m in mids:
        smis, keys, _, scs, forms = state['base'][m][:5]
        tk = cp.canonicalize_smiles_to_ik14(T[m]['smiles'])
        hit = np.array([cp.canonicalize_smiles_to_ik14(s) == tk for s in smis], bool)
        if not len(smis):
            for k in res: res[k].append(0)
            continue
        isg = np.array([f.endswith('|gen') for f in forms])
        pid = np.array([key2pid.get(k, -1) for k in keys])
        assert not ((pid < 0) & ~isg).any(), (shard, m)
        p = np.where(isg, 0.0, pop[np.maximum(pid, 0)])
        z = np.asarray(scs, np.float64) - 0.15 * p
        pool_i = np.flatnonzero(~isg)

        def first(score):
            o = np.empty(len(smis), np.int64); o[isg] = np.flatnonzero(isg)
            o[~isg] = pool_i[np.argsort(-score[pool_i], kind='stable')]
            r = np.flatnonzero(hit[o])
            return int(r[0]) + 1 if len(r) else 0
        res['as_run'].append(int(np.flatnonzero(hit)[0]) + 1 if hit.any() else 0)
        for mu in (0.0, 0.075, 0.30):
            res[mu].append(first(z + mu * p))
        res['pop_only'].append(first(p + 1e-6 * z))
    print(f'== {shard} (n {len(mids)})  BASE top-1 / MRR@60 by popularity weight MU (run used 0.15)')
    for k, v in res.items():
        v = np.array(v); rr = np.where(v > 0, 1.0 / np.maximum(v, 1), 0.0)
        print(f'   MU {str(k):>8}: top1 {np.mean(v == 1):.3f}  mrr {rr.mean():.3f}')
