"""c3-iso I1b.1 measure (Kaggle CPU, internet on for rdkit==2026.3.3 = the metric's tautomer-canonical IK14).
Inputs: K1 output (ice_out_pseudo_c3.json, ice_run_summary.json), the I1b.0 frozen pkls + truths (measure dataset),
cand_meta.json. Truth is loaded only AFTER the ICE scores exist on disk (they come from a finished kernel) and the
per-molecule candidate ranking is computed without it; truth only reports reach.

Children ranking by ICE: score desc; None (unscored) after all scored, in CPU-frag order. A molecule with no covered spectrum has
no ICE score: reported separately (ice-ranked denominator = ICE-covered molecules; all-600 denominator with frag-order fallback)."""
import glob, json, os, pickle, subprocess, sys, time, math
from multiprocessing import Pool
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rdkit==2026.3.3"], check=False)
import rdkit; print("rdkit", rdkit.__version__, flush=True)
def one(pat):
    h = sorted(glob.glob(pat, recursive=True), key=len); assert h, pat; return h[0]
SRC = os.path.dirname(one("/kaggle/input/**/truths_pseudo_c3.pkl"))
OUTD = os.path.dirname(one("/kaggle/input/**/ice_out_pseudo_c3.json"))
META = one("/kaggle/input/**/cand_meta.json")
sys.path.insert(0, SRC)
import casmi_post as cp
ice = json.load(open(f"{OUTD}/ice_out_pseudo_c3.json"))
run = json.load(open(f"{OUTD}/ice_run_summary.json"))
meta = json.load(open(META))
fr = pickle.load(open(f"{SRC}/i1b_pseudo_c3_full_top150.pkl", "rb"))
print("ICE scores on disk for", len(ice), "molecules; frozen top150 for", len(fr["top"]), flush=True)
T = pickle.load(open(f"{SRC}/truths_pseudo_c3.pkl", "rb"))   # truth loaded here, after the scores exist
smis = {t["smiles"] for t in T.values()}
for m, cm in meta.items():
    smis.update(cm)
smis = sorted(s for s in smis if isinstance(s, str))
t0 = time.time()
with Pool(4) as p:
    keys = p.map(cp.canonicalize_smiles_to_ik14, smis, chunksize=200)
cache = dict(zip(smis, keys)); print(f"tautomer ik14 {time.time()-t0:.0f}s for {len(smis)}", flush=True)

def wilson(k, n, z=1.6449):
    if n == 0: return [None, None]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]
def rate(fl):
    n, k = len(fl), int(sum(fl)); return dict(k=k, n=n, rate=(k / n if n else None), ci90=wilson(k, n))

rows = []
for m, cm in meta.items():
    sc = ice.get(m, {})
    kids = [(s, v[1], v[2]) for s, v in cm.items() if v[0] == "child"]        # (smiles, op, frag), frag order already
    fused = [s for s, v in cm.items() if v[0] == "fused"]
    covered = any(sc.get(s) is not None for s in cm)
    order = sorted(range(len(kids)), key=lambda i: (sc.get(kids[i][0]) is None, -(sc.get(kids[i][0]) or 0.0), i))
    tk = cache[T[m]["smiles"]]
    ktaut = [cache.get(kids[i][0]) for i in order]
    hit = [j + 1 for j, k in enumerate(ktaut) if k == tk]
    frag_hit = [j + 1 for j, k in enumerate(cache.get(s) for s, _, _ in kids) if k == tk]
    best_kid = max((sc.get(k[0]) for k in kids if sc.get(k[0]) is not None), default=None)
    best_fus = max((sc.get(s) for s in fused if sc.get(s) is not None), default=None)
    r = hit[0] if hit else 0
    rows.append(dict(mid=m, covered=covered, n_kids=len(kids), n_kids_scored=sum(sc.get(k[0]) is not None for k in kids),
                     ice_rank=r, frag_rank=(frag_hit[0] if frag_hit else 0),
                     op=(kids[order[r - 1]][1] if r else None),
                     reach_20=0 < r <= 20, reach_50=0 < r <= 50, reach_150=r > 0,
                     frag_reach_20=bool(frag_hit) and frag_hit[0] <= 20,
                     truth_in_fused10=any(cache.get(s) == tk for s in fused),
                     truth_ice=(sc.get(kids[order[r - 1]][0]) if r else None), best_kid=best_kid, best_fused=best_fus,
                     kid_beats_fused=(best_kid is not None and best_fus is not None and best_kid > best_fus)))
cov = [r for r in rows if r["covered"]]
def block(rs):
    return {k: rate([r[k] for r in rs]) for k in ("reach_20", "reach_50", "reach_150", "frag_reach_20")}
ops = {}
for r in rows:
    if r["reach_20"]: ops[r["op"]] = ops.get(r["op"], 0) + 1
ranks = sorted(r["ice_rank"] for r in rows if r["ice_rank"])
summ = dict(n=len(rows), n_ice_covered=len(cov), all_600_frag_fallback=block(rows), ice_covered=block(cov),
            reach_20_by_ice_all600=rate([r["reach_20"] for r in rows]),
            not_in_fused10=block([r for r in rows if not r["truth_in_fused10"]]),
            hit_ops_ice20=ops, truth_ice_rank_median=(ranks[len(ranks) // 2] if ranks else None),
            truth_ice_rank_hist=dict(le5=sum(x <= 5 for x in ranks), le20=sum(x <= 20 for x in ranks),
                                     le50=sum(x <= 50 for x in ranks), le150=len(ranks)),
            kid_beats_fused=dict(all=sum(r["kid_beats_fused"] for r in rows),
                                 on_truth_hit20=sum(r["kid_beats_fused"] for r in rows if r["reach_20"]),
                                 of_covered=len(cov)),
            ice_run=run, rdkit=rdkit.__version__)
json.dump(dict(summary=summ, rows=rows), open("/kaggle/working/i1b1_measure.json", "w"), indent=1, default=str)
json.dump(summ, open("/kaggle/working/i1b1_summary.json", "w"), indent=1, default=str)
print(json.dumps(summ, indent=1, default=str))
