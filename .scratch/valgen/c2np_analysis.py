"""Option A C2_np offline checks: H0 replay == submission.csv, channel rates, Engine 2 rank-given-reachable vs C2,
and every back-test flip with USE_ENG on/off. Imports research/harness/harness.py (read-only use)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

H = Path('C:/Users/h/OneDrive/Desktop/ENVEDA_casmi/research/harness')
sys.path.insert(0, str(H))
import harness as HX  # noqa: E402

shard = sys.argv[1] if len(sys.argv) > 1 else 'C2_np'
T = HX.truths()
res = {}


def chan(shard):
    state, deps, cp = HX.load_dump(shard)
    mids = [m for m in state['te_mids'] if m in T]
    tk = {m: cp.canonicalize_smiles_to_ik14(T[m]['smiles']) for m in mids}
    ch = HX._channels(state, mids, tk, cp)

    def pos(lst, t):
        return next((i for i, s in enumerate(lst[:60], 1) if cp.canonicalize_smiles_to_ik14(s) == t), 0)
    e = np.array([pos(((state.get('eng') or {}).get(m) or {}).get('smiles') or [], tk[m]) for m in mids])
    reach = e[e > 0]
    ch['Engine 2 | reachable'] = dict(n=int(len(reach)), top1=float(np.mean(reach == 1)) if len(reach) else None,
                                     mrr=float(np.mean(1.0 / reach)) if len(reach) else None,
                                     median_rank=float(np.median(reach)) if len(reach) else None)
    out = cp.run(state, deps)
    rk = np.array([HX.rank_of(out.get(m, []), tk[m], cp) for m in mids])
    ch['final'] = dict(top1=float(np.mean(rk == 1)), mrr=float(HX.rr(rk).mean()), n=len(mids))
    return state, deps, cp, mids, tk, out, ch


state, deps, cp, mids, tk, out, ch = chan(shard)
sub = pd.read_csv(H / 'dumps' / shard / 'submission.csv')
bad = [m for m, s in zip(sub.molecule_id.astype(str), sub.smiles) if ';'.join(out.get(m, ['<missing>'])) != str(s)]
res['H0'] = dict(n=len(sub), mismatches=len(bad), examples=bad[:5])
res['channels'] = {shard: ch}
if '--c2' in sys.argv:
    res['channels']['C2'] = chan('C2')[-1]
spec = HX.load_spec()
fl = []
for f in spec['flips']:
    row = dict(id=f['id'], set=f['set'], tier=f.get('tier'), lb=f.get('lb'))
    for eng in (True, False):
        a = {} if eng else {'USE_ENG': False}
        b = dict(f['set']) if eng else dict(f['set'], USE_ENG=False)
        if not eng and f['set'] == {'USE_ENG': False}:
            continue
        r = HX.compare(shard, a, b)
        row['eng_on' if eng else 'eng_off'] = dict(delta=r['delta'], ci90=r['ci90'], up=r['up'], down=r['down'])
    fl.append(row)
res['flips'] = fl
print(json.dumps(res, indent=1, default=str))
json.dump(res, open(H / 'dumps' / shard / 'option_a_checks.json', 'w'), indent=1, default=str)
