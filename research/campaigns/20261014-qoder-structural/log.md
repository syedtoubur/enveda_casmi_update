# Campaign log: 20261014-qoder-structural

- 2026-10-14 00:40 | charter | - | charter.md, state.json written (Qoder, tier M); battery on disk: C2, C1_npx, C2_hard, C1_xlib, C2_np (advisory)
- 2026-10-14 00:55 | replan | research/exploit/20261014-refspectra-lib | pick refspectra-lib (only measured non-zero ceiling; xattn needs GPU training + approval; formula-window reach 10 mol). EV: C1_npx 17 targets, LB <= +0.004 assumed, cost 5
- 2026-10-14 01:22 | spike | research/exploit/20261014-refspectra-lib | NO SIGNAL on leak-free molecules (dother=0 every cohort after self-copy cut; GNPS/MoNA 0/6, 3 rank-1 displacements) -> spike_result.md (Qoder)
- 2026-10-14 handover | Qoder stopped mid-run; Claude resumes. Human: close refspectra-lib without LB probe (STATUS needs-human); drive the rest with campaign.py --once. Submit autonomy OFF.
- 2026-10-14 06:22 cycle 1 closed: 20261014-refspectra-lib -> needs-human
- 2026-10-14 06:22 report (sonnet) brief 7874 chars  OVER 6000
- 2026-10-14 06:23 replan (opus) brief 7004 chars  OVER 6000
- 2026-10-14 06:30 | replan | research/exploit/20261014-formula-window | cycle 1 taught: refspectra-lib's reach was all leak-hit or self-copy (dother=0), so it closed without an LB probe. Pick formula-window (branch candidate-recall), the only cheap live idea. xattn-rerank needs GPU training plus human approval, and its cheap form MEASURED -0.0218. EV: reach 2.5% C2 x +0.007 ASSUMED x P 0.4 / cost 2 = 0.0014 (xattn 0.0003). Kill early if the posterior's top 3 holds the truth formula on < 4 of the 10 targets.
- 2026-10-14 06:24 spec (opus) brief 7648 chars  OVER 6000
- 2026-10-14 06:45 cycle 2 closed: 20261014-formula-window -> needs-human
- 2026-10-14 06:45 | human | - | proposals P1 approved (L43); formula-window dropped (needs-human); human seeds ideation with a Class-1 library-gate diagnosis (LIB_TAU 0.9 vs cross-instrument eval; smart gate row 55 -0.011). Replan waits for research/ideas/2026-10-14-libgate-diagnosis.md
- 2026-10-14 06:50 | diagnosis | - | libgate diag (.scratch/libgate_diag, pre-reg gate_report 2026-10-14): 0.9 does reject 83% of true cross-instrument C1 matches, but timsTOF Class-2 queries false-lock 24.5% at 0.7; C1 loss is isomer order (BASE wrong, truth rank 2), not the threshold; ice-libprotect @0.9 C1_npx +0.0054 [+0.001,+0.011], C2 neutral (lead, small). -> research/ideas/2026-10-14-libgate-diagnosis.md; state.ideate set
- 2026-10-14 06:50 report (sonnet) brief 7405 chars  OVER 6000
- 2026-10-14 06:51 diagnose (sonnet) brief 6410 chars  OVER 6000
- 2026-10-14 06:52 ideate (opus) brief 6909 chars  OVER 6000
- 2026-10-14 06:56 ideation round 1 done; back to /replan
- 2026-10-14 06:56 fetch (haiku) brief 6729 chars  OVER 6000
- 2026-10-14 06:58 replan (opus) brief 7004 chars  OVER 6000
- 2026-10-14 07:05 | replan | research/exploit/20261014-lib-isomer-margin | cycles taught: refspectra-lib (cycle 1) had all its reach on leak-hit or self-copy molecules (dother=0); formula-window (cycle 2) was reach-starved and dropped by the human. Pick candidate A, lib-isomer-margin (branch scorer, diagnosis H1): a comparative per-candidate library score among library-backed top-3 isomers on C1_npx/C1_xlib, read on d_lib_leak. EV: +0.005 ASSUMED on a MEASURED bucket (C1_npx -0.052) x P 0.4 / cost 2 = 0.0010 (B 0.00075; C 0.00125, but a below-noise knob kept as stack partner with D). Kill: truth's own-ref sim beats the rank-1 isomer's in < 50% of the 17 rank-2 cases, or C2_np false-swap > 2%.
- 2026-10-14 07:00 spec (opus) brief 7660 chars  OVER 6000
- 2026-10-14 07:02 preflight 20261014-lib-isomer-margin: 0 finding(s)
- 2026-10-14 07:02 attack (opus) brief 7531 chars  OVER 6000
- 2026-10-14 08:16 | build (manual, human-directed) | research/exploit/20261014-lib-isomer-margin | spec/preflight/attack done; falsifier kernel + build notebook (casmi26-e006-lib-isomer.ipynb, branch idea/lib-isomer-margin) built by two Haiku subagents, reviewed, tests green; harness RDKit preseed fixed (tag-matched wheel). Human approved dataset+2 pushes in a fresh session: .scratch/handoffs/20261014-0815-lib-isomer-ship.md. Do not step campaign.py before the falsifier verdict.
- 2026-10-14 09:41 | parallel (manual) | research/exploit/20261014-ice-margin-route | DEAD: C1_npx -0.0007 [-0.0021, +0.0000] <= 0 (pre-reg kill bar <= +0.0054); C1_xlib +0.0053, C2 -0.0005, C2_hard +0.0015 | outside the driver, worktree exploit-b

- 2026-10-14 14:19 stopped: human (grill 2026-10-14): superseded by 20261014-e180-npool-mutator; lib-isomer-margin already submitted, bind its LB with poll_lb.py --claim
