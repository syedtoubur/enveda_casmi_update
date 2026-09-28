# =====================================================================================
#  THE ONE FAIR TEST STAGE D CAN STILL DO.  Paste into the LIVE Stage-D kernel.
#
#  The new FPNet's training exclusion is the union of every val molecule's ban_keys --
#  15,194 structures, of which 14,872 (97.9%) come from C2_hard's Murcko-scaffold-family
#  bans.  C2 and C2_hard truths are 100% inside that union, so no fair subset of them
#  exists and no amount of re-running Stage D can build one.
#
#  BUT: 38 of 245 C1_npx truths and 30 of 300 C1_xlib truths also landed inside the union,
#  swept up by some C2_hard molecule's family ban.  Those 68 molecules are Class-1 queries
#  whose chemistry the new model was DENIED -- the same handicap C2 suffers, but measured
#  where the library channel still works and where `base` holds no memorisation edge over
#  a structure it also has.
#
#  If `both` holds up on the denied 68 as well as on the other 477, the C2 deficit is a
#  property of the handicap, not of the model.  If `both` collapses on the denied 68, the
#  new FPNet really is weak on unseen chemistry and the C2 result should be believed.
# =====================================================================================
import numpy as np
import pandas as pd

BANU = set()
for r in key:
    BANU.update(r['ban_keys'])
TRU = pd.Series({r['molecule_id']: r['inchikey14'] for r in key})
print(f"exclusion union: {len(BANU):,} structures")

b = RES['base'].set_index('mid')
denied = TRU.reindex(b.index).isin(BANU).values

print(f"\n{'cohort':9s} {'truth denied':13s} {'n':>5s} "
      f"{'base MRR':>9s} {'both MRR':>9s} {'delta':>9s} {'SE':>8s}")
for coh in ['C1_npx', 'C1_xlib', 'C2', 'C2_hard']:
    for want in (True, False):
        m = (b.cohort == coh).values & (denied == want)
        if m.sum() < 5: continue
        bb = RES['both'].set_index('mid')
        d = (bb.rr - b.rr)[m]
        print(f"{coh:9s} {str(want):13s} {m.sum():5d} "
              f"{b.rr[m].mean():9.4f} {bb.rr[m].mean():9.4f} "
              f"{d.mean():+9.4f} {d.std()/np.sqrt(len(d)):8.4f}")
    print()

# the headline: all Class-1 molecules, split by whether the new model was denied them
bb = RES['both'].set_index('mid')
c1 = b.cohort.isin(['C1_npx', 'C1_xlib']).values
for want, lab in ((True, 'DENIED to new FPNet'), (False, 'available to new FPNet')):
    m = c1 & (denied == want)
    d = (bb.rr - b.rr)[m]
    print(f"Class-1, {lab:24s} n={m.sum():4d}  delta {d.mean():+.4f} "
          f"+- {d.std()/np.sqrt(len(d)):.4f}")
print()
print("Read it: if the two lines are close, the new FPNet handles denied chemistry fine and")
print("the C2 deficit is the handicap talking. If the DENIED line is much worse, believe C2.")
