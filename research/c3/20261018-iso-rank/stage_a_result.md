# iso-rank Stage A result

**Verdict: DEAD** (pre-registered: DEAD iff Delta_pseudo_c3 on half B < 0.012). Delta_pseudo_c3 (half B, n=302) = **+0.0015**, CI90 [+0.0005, +0.0024].

Kernels (Kaggle CPU, private): `c3iso-stagea-a3-s0` v2, `a3-s1` v2, `a3-pseudo` v2, `a4-freeze` v1, `a4-score` v1; dataset `casmi26-c3iso-stagea-in` (+ kit, validation-data). Code: worktree c3-iso-rank-a, `stage_a/`.
A3 coverage: C1_npx 245, C1_xlib 300, C2 400, C2_hard 200 (all of val_key), pseudo_c3 600 (719 of ~90k frozen kids had no regenerated edge).

## Frozen on half A (committed edb521a before half B was scored)
lists = ice, mult, parent_rank, frag, nl (all five available; forward selection kept every list), m_ice = 0.02, half A n=298, within-pool MRR@20 0.319 (pool 49). Half-A Delta grid: m 0: 0.0043, 0.01: 0.0049, 0.02: 0.0050, 0.05: 0.0024, 0.1: 0.0033.

## Half B pseudo_c3 (frozen config)
Delta +0.0015 [+0.0005, +0.0024] (scaffold bootstrap), 12 up / 2 down; fire rate 21.9%; tail fills 99.7% of molecules. Fire-only (no tail) Delta -0.0003: the gain comes from the tail ride-along, not from the parent-anchored insert. Acyclic +0.0016 (n=56), ring +0.0015 (n=246); timsTOF 0.0 (n=6; only 6 molecules). Half A reference Delta +0.0050. truth in MSG: not measurable (msg_ik14.npy only on Colab Drive).
Fast-ship config (all five lists, m_ice 0, no GLACIER): half B Delta +0.0013 [+0.0004, +0.0023], fire rate 32.1%, fire-only -0.0005. Final-row-parents variant: identical by construction (supplementary, not used for any choice).

## C1/C2 half B, frag-only (ICE conjunct dropped, so fire rates are an upper bound)
| cohort | n | fire rate | Delta (actual = worst case) |
|---|---|---|---|
| C1_npx | 124 | 0.84 | -0.0045 |
| C1_xlib | 154 | 0.75 | -0.0050 |
| C2 | 182 | 0.64 | -0.0209 |
| C2_hard | 100 | 0.67 | -0.0165 |
Fire-only worst case: -0.0042, -0.0050, -0.0206, -0.0165. Children never matched a C1/C2 truth, so actual = worst case.
Composite bracket (equal C1/C2 split of the 1-pi share): pi 0.15: -0.0097; 0.25: -0.0084; 0.35: -0.0071 (worst case = actual). The fast-ship numbers are within 0.0001.

## Caveats
No ICE or GLACIER on C1/C2 (Stage B would add the ICE conjunct, lowering fire rates); C1/C2 frag-only fire rates are therefore not the Stage B quantity. Even so the pseudo_c3 gain (0.0015, and 0.005 on the selection half) is an order of magnitude under the 0.012 bar and the composite is negative.
