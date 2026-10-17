"""GEN1_KEEP per-cohort report: fires, gen1_moved, kept-correct, rank-1 displacements, leak-hit split.
Usage: python gen1_report.py <out.json> [post.py]   (run from E:/ENVEDA_casmi; reads harness dumps; read-only)"""
import json, sys
from pathlib import Path
sys.path.insert(0, 'E:/ENVEDA_casmi/research/harness')
import harness as h

V22 = dict(METRIC_FUSION=True, ICE_FALLBACK=True, FUSION_WINDOW=40, REL_PROMOTE=150.0, FRAG_GAP=True, E2R2=True)
SHARDS = ['pseudo_c3', 'C2', 'C1_npx', 'C2_hard', 'C1_xlib']


def main():
    outp = sys.argv[1]
    post = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    T = h.truths()
    res = {}
    for shard in SHARDS:
        state, deps, cp = h.load_dump(shard, post)
        mids = [m for m in state['te_mids'] if m in T]
        tk = {m: cp.canonicalize_smiles_to_ik14(T[m]['smiles']) for m in mids}
        st = {}
        a = cp.run(state, deps, **V22)
        b = cp.run(state, deps, stats=st, GEN1_KEEP=True, **V22)
        r = dict(n=len(mids), changed_rows=sum(a[m] != b[m] for m in a), gen_base1=0, gen_base1_correct=0,
                 gen_correct_rank1_v22=0, gen_correct_rank1_b=0, fires=st.get('gen1_kept', 0),
                 gen1_moved=st.get('gen1_moved', 0),
                 displaced_rank1=0, gained_rank1=0)
        for m in mids:
            smis, _k, _l, _s, forms = state['base'][m]
            ra = h.rank_of(a[m], tk[m], cp); rb = h.rank_of(b[m], tk[m], cp)
            is_gen = bool(forms) and str(forms[0]).endswith('|gen')
            corr = is_gen and cp.canonicalize_smiles_to_ik14(smis[0]) == tk[m]
            r['gen_base1'] += is_gen; r['gen_base1_correct'] += corr
            if corr:
                r['gen_correct_rank1_v22'] += ra == 1; r['gen_correct_rank1_b'] += rb == 1
            if a[m] != b[m]:
                if ra == 1 and rb != 1: r['displaced_rank1'] += 1
                if rb == 1 and ra != 1: r['gained_rank1'] += 1
        r['displaced_pct'] = 100.0 * r['displaced_rank1'] / max(1, r['n'])
        res[shard] = r
        print(shard, r, flush=True)
    Path(outp).write_text(json.dumps(res, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
