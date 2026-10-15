"""C2_np BASE-leak diagnosis, probes P1 + P2 (CPU, read-only).
P1: train rows that survive the native-ik14 ban but are the truth under RDKit's plain IK14.
P2: for BASE top-1 == truth (score key), is the top-1 SMILES the exact truth (plain IK14) or a variant; lib_max."""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

R = Path('E:/ENVEDA_casmi')
sys.path.insert(0, str(R / 'research/harness'))
import harness as HX  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402
RDLogger.DisableLog('rdApp.*')

state, deps, cp = HX.load_dump('C2_np')
T = HX.truths()
mids = [m for m in state['te_mids'] if m in T]
truth_ik = {m: T[m]['inchikey14'] for m in mids}

tm = pd.read_parquet(R / '.scratch/valgen/train_meta.parquet', columns=['ik14', 'ik14_rdkit', 'lib'])
tset = set(truth_ik.values())
# P1
surv = tm[tm.ik14_rdkit.isin(tset) & ~tm.ik14.isin(tset)]
print(f'P1 rows surviving the native ban but RDKit-ik14 == truth: {len(surv)} rows, '
      f'{surv.ik14_rdkit.nunique()} truths; libs {dict(Counter(surv.lib))}')
train_ik = set(tm.ik14.values) - tset          # what remains in train after the ban (native keys)
train_ik |= set(tm.ik14_rdkit[~tm.ik14.isin(tset)].values)
del tm


def plain(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToInchiKey(m)[:14] if m is not None else None


rows = []
for m in mids:
    smis, keys, lib_max = state['base'][m][0], state['base'][m][1], state['base'][m][2]
    tk = cp.canonicalize_smiles_to_ik14(T[m]['smiles'])
    top1 = bool(smis) and cp.canonicalize_smiles_to_ik14(smis[0]) == tk
    p = plain(smis[0]) if smis else None
    rows.append(dict(mid=m, top1=top1, lib_max=float(lib_max), exact=(p == truth_ik[m]),
                     top1_in_train=(p in train_ik), n=len(smis)))
d = pd.DataFrame(rows)
h = d[d.top1]
print(f'P2 n={len(d)} BASE top1==truth {d.top1.mean():.3f}')
print(f'   among hits: exact truth structure {h.exact.mean():.3f}, variant {(~h.exact).mean():.3f}, '
      f'variant present in banned train {(~h.exact & h.top1_in_train).sum()}')
print('   lib_max quantiles hits  ', np.round(h.lib_max.quantile([.1, .25, .5, .75, .9]).values, 3))
print('   lib_max quantiles misses', np.round(d[~d.top1].lib_max.quantile([.1, .25, .5, .75, .9]).values, 3))
print('   hit rate by lib_max bin', d.groupby(pd.cut(d.lib_max, [-.01, .2, .4, .6, .8, 1.01]), observed=True).top1.agg(['mean', 'size']).to_dict('index'))
d.to_csv(R / 'research/harness/dumps/C2_np/leak_probe1.csv', index=False)
