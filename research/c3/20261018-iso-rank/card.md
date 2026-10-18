# Submit card: iso-rank fast-ship (c3-iso campaign)

- Kernel: `syedtoubur/casmi26-iso-rank` v1 (private, T4, internet off, control's pinned docker image, +`syedtoubur/casmi26-e180-keys`). Code: worktree E:/casmi/lines/c3-iso-rank commit fd81842, K = 150, fast-ship cfg (m_ice 0, k 10, six lists, tail on) frozen in gate_report `c3-iso iso-rank, PRE-REGISTERED`.
- What it does: V22 + one final cell. Isomer children of the final top 10, ranked by truth-blind RRF (ICE, frag, multiplicity, parent rank, NL, GLACIER); best child inserted directly below its parent only if it beats it on ICE and frag; ranks 21-25 filled from the RRF list.
- Control: V22 LB 0.447 (ref 57306053), unpaired (flag). gen-rank1-keep scored 0.446 (same base).
- KEEP iff LB >= 0.451 AND hidden counters in band: iso_fired in 2-30 % of molecules, iso_children > 0 on >= 90 %. If LB in 0.445-0.453, push one fresh V22 replicate before deciding.
- Draft counters (12 molecules): `ISO_COUNTERS iso_children=12 iso_fired=4 iso_tail=12 iso_skipped_time=0 n_mol=12`; ice/gl covered 11 of 12. Fire rate 4/12 = 33 % on 12 molecules is above the 30 % hidden band (tiny sample; judge the band on the hidden counters).
- Runtime projection (runtime.md): K=150 about 8.0 h pessimistic (6.5 h at the draft rates), 8.3 h guard. Hidden re-run gives the real number.
- Flags: no offline measurement beyond I1b.1 reach; Stage A/B not waited for (fast-ship); truth-blind; LB-vs-offline disagreement is a calibration datapoint.
- G4: PASS (see gates.md).

## WITHDRAWN 2026-10-18 (not submitted)
Stage A DEAD (pseudo_c3 half B +0.0015 < 0.012; fire-only -0.0003) and the C1/C2 frag-only composite at pi 0.25 is -0.0084
(`stage_a_result.md`). The fast-ship probe costs an LB slot with negative expected value, so it is withdrawn.
Follow-up: `research/c3/20261018-signal-audit/` (per-signal C1/C2 false-alarm audit, then K2 v1 detector).
