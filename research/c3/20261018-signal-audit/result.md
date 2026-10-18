# c3 signal audit: result

**Verdict: ALL DEAD** (pre-registered gate_report 2026-10-18 "c3 signal audit"; sanity PASS). No CPU child-vs-parent
signal comes near the 4.5 % false-alarm bar; none fires more often on Class 3 than on C1/C2. Placing a child above its
parent with these signals is closed for good (rule R11).

Source: `kaggle_output/audit_result.json`, `audit_rows.csv`, `audit.log` (kernel `syedtoubur/c3iso-signal-audit` v1,
Kaggle CPU; code `E:/casmi/lines/c3-signal-audit` @ 1a09587).

## Sanity (void unless ±0.005)
Frag-only half-B fire rates reproduce Stage A: C1_npx 0.839 (ref 0.84), C1_xlib 0.753 (0.75), C2 0.643 (0.64), C2_hard 0.670 (0.67).

## Verdicts (FPR pooled over all 1,145 C1/C2 molecules; TPR on pseudo_c3 half A, n 298)
| rule | C1/C2 FPR | C1/C2 "other" FPR (n 133) | pseudo_c3 A fire | TPR (fire ∧ c* = truth) | net = 0.25·TPR − 0.275·FPR | P(pos(p)=1 \| fire) |
|---|---|---|---|---|---|---|
| S_nl (dnl > 0) | 0.473 | 0.489 | 0.450 | 0.000 | −0.130 | 0.48 |
| S_mult (mult ≥ 2) | 0.528 | 0.504 | 0.379 | 0.017 | −0.141 | 0.68 |
| S_frag (dfrag > 0) | 0.707 | 0.662 | 0.711 | 0.003 | −0.194 | 0.52 |
| Rule A (dnl > 0 ∧ mult ≥ 2) | 0.222 | 0.241 | 0.134 | 0.000 | −0.061 | 0.68 |
| Rule B (mult ≥ 3 ∧ dfrag > 0) | 0.214 | 0.188 | 0.111 | 0.003 | −0.058 | 0.80 |

Per cohort (all): Rule A fires on C1_npx 0.40, C1_xlib 0.23, C2 0.15, C2_hard 0.14; S_nl 0.59 / 0.46 / 0.44 / 0.42.

## Findings
- **No discrimination at any threshold.** C1/C2 vs pseudo_c3-A fire curves: dnl > 0.1: 0.079 vs 0.067; dnl > 0.2: 0.027 vs 0.027;
  mult ≥ 4: 0.193 vs 0.064; dfrag > 0.1: 0.078 vs 0.131. mult fires *more* on C1/C2 (library compounds have dense isomer
  neighbourhoods). Only dfrag leans toward C3, and weakly (LR ≈ 1.7 at > 0.1, with TPR ≈ 0).
- **nl is tie-dominated, as the referee suspected.** dnl = 0 on 46 % of C1/C2 and 48 % of pseudo_c3-A molecules; both zero
  on 9 % / 8 %. "The 4–8 % false alarms" were never real: S_nl fires on 47 %.
- **The ceiling binds first.** With the CPU lists, c* is the truth on 2.0 % of half A (6/298) and 1.0 % of half B. Even
  a perfect veto could add at most 0.25 × 0.02 = +0.005 at π 0.25.
- **Half-B `delta_fire_only == delta_wc` explained:** the frozen (ICE) config fired 66 times on half B and **0** inserted
  children were the truth (half A: 68 fires, 4 true). Not a wiring bug.
- **Tail-only ride-along (21–25, no fire):** C1/C2 Δ −0.0002 / 0.0 / −0.0001 / −0.0002 (all), "other" −0.0018 / 0.0 / −0.0012 /
  −0.0014; 1 truth demoted per cohort at most. pseudo_c3 +0.0029 (A), +0.0016 (B). Composite at π 0.25 ≈ +0.0004: real
  but far below the ±0.004 LB noise floor, so not an LB slot on its own.

## Next (pre-registered branch)
All DEAD → K2 v1 detector spec (plan Step 3). No pseudo_c3_v2 draw, no GPU from this audit.
