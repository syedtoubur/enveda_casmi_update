# =====================================================================================
#  STAGE-D DIAGNOSTIC -- paste into the LIVE Stage-D kernel, in a new cell at the bottom.
#
#  Stage D said HOLD: `both` loses 0.044 on C2 and 0.040 on C2_hard. Before accepting that
#  as a verdict on the new stack, two properties of those cohorts have to be ruled out,
#  because both of them bias the comparison in `base`'s favour and neither exists on the
#  leaderboard:
#
#   1. COLLISION ENERGY. C2 is 77% ce_known=False and C2_hard is 85%. The test set always
#      reports CE in eV, so those molecules are scored through FPNet's 25.0 eV fallback,
#      not the path either model was trained for. Stage C deliberately preferred CE-known
#      rows -- so the new matrix was built for a regime C2 mostly does not exercise.
#   2. TRAINING-SET ASYMMETRY. The new FPNet was denied all 15,194 val-banned structures by
#      design. The old one, trained by the fork's author, had no reason to exclude them.
#      In C1 the library channel masks that; in C2 the model channel is the only structural
#      evidence there is, so any memorisation advantage lands exactly there.
#
#  Part A splits every paired delta by ce_known. Part B measures what each FPNet actually
#  does in-window, per cohort -- the number that separates "our model is worse" from "their
#  model memorised this val set".
# =====================================================================================
import numpy as np
import pandas as pd

CEK = pd.Series({r['molecule_id']: bool(r['ce_known']) for r in key})
COH = ['C1_npx', 'C1_xlib', 'C2', 'C2_hard']

print("=" * 86)
print("A. PAIRED DELTAS vs base, SPLIT BY ce_known   (the test set is 100% ce_known=True)")
print("=" * 86)
b = RES['base'].set_index('mid')
ck = CEK.reindex(b.index).fillna(False).astype(bool)
print(f"{'config':12s} {'cohort':9s} {'ce_known':9s} {'n':>5s} {'delta':>9s} {'paired SE':>10s}")
for name in ('model_only', 'matrix_only', 'both'):
    d = RES[name].set_index('mid').rr - b.rr
    for coh in COH:
        for want in (True, False):
            m = (b.cohort == coh).values & (ck.values == want)
            if m.sum() == 0: continue
            v = d[m]
            se = v.std() / np.sqrt(len(v))
            flag = ""
            if abs(v.mean()) > se: flag = "  <-- above 1 SE"
            print(f"{name:12s} {coh:9s} {str(want):9s} {m.sum():5d} "
                  f"{v.mean():+9.4f} {se:10.4f}{flag}")
    print()

print("=" * 86)
print("B. WHAT EACH FPNet DOES IN-WINDOW, per cohort")
print("=" * 86)
if 'ZLOG' not in globals():
    print("  ZLOG is gone (kernel restarted) - re-run pass 2, or skip: part A alone is")
    print("  enough to judge the collision-energy confound.")
else:
    # column 20 of rank_features is the indicator 'this row is the model's top-1 in its
    # window'. Reading it through rank_features itself keeps this identical to scoring.
    rec = []
    for mid, _ in mols:
        ent = CACHE.get(mid)
        if ent is None or not len(ent['cand']): continue
        ti = np.flatnonzero(pool.keys[ent['cand']] == TRUTH[mid])
        if not len(ti): continue                      # truth outside the window: no opinion
        cfp = pool.fps(ent['cand'])
        afp = pool.fps(ent['ids']) if len(ent['ids']) else None
        row = dict(mid=mid, cohort=COHORT[mid], ce=bool(CEK.get(mid, False)),
                   nwin=len(ent['cand']))
        for tag in ('old', 'new'):
            z = ZLOG[tag].get(mid)
            if z is None: continue
            X = rank_features(cfp, ent['lv'], afp, ent['sims'], z, ent['fsc'])[:, :NFEAT]
            row[tag] = float(X[ti[0], 20])            # truth is the model's top-1?
        rec.append(row)
    D = pd.DataFrame(rec)
    print(f"  truth reachable in-window for {len(D):,}/{len(mols):,} molecules\n")
    print(f"{'cohort':9s} {'ce_known':9s} {'n':>5s} {'window':>7s} {'OLD top-1':>10s} "
          f"{'NEW top-1':>10s} {'NEW-OLD':>9s}")
    for coh in COH:
        for want in (True, False):
            s = D[(D.cohort == coh) & (D.ce == want)]
            if len(s) < 5: continue
            print(f"{coh:9s} {str(want):9s} {len(s):5d} {s.nwin.mean():7.0f} "
                  f"{s.old.mean()*100:9.1f}% {s.new.mean()*100:9.1f}% "
                  f"{(s.new.mean()-s.old.mean())*100:+8.1f}")
        s = D[D.cohort == coh]
        print(f"{coh:9s} {'ALL':9s} {len(s):5d} {s.nwin.mean():7.0f} "
              f"{s.old.mean()*100:9.1f}% {s.new.mean()*100:9.1f}% "
              f"{(s.new.mean()-s.old.mean())*100:+8.1f}")
        print()
    print("  Reference points measured elsewhere:")
    print("    new FPNet, scaffold-disjoint holdout vs 48 real decoys : es_top1 74.2%")
    print("    new FPNet, Stage-C windows (~70 cand), UNSEEN structures:         52.1%")
    print("    new FPNet, Stage-C windows, SEEN (memorised) structures :         91.7%")
    print("    old FPNet, forked matrix windows                        :         42.0%")
    print()
    print("  If OLD beats NEW on C2/C2_hard while NEW is at ~52% (its honest rate), the old")
    print("  model is reading structures it was trained on -- an advantage it will not have")
    print("  on the test set, where no Class-2 molecule is in anyone's training data.")
    D.to_csv('/content/drive/MyDrive/casmi_2026/_staged/stage_d_model_top1.csv', index=False)
    print("\n  per-molecule -> _staged/stage_d_model_top1.csv")
