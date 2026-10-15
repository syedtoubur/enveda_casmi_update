
## 2026-10-14: C1 "other Δ" hides library-driven gains (loop proposal; human decides, not applied)
- On C1 cohorts, "leak-hit" (BASE rank 1 = truth, `harness.py:267,446`) is mostly a genuine library match, not the
  FPNet leak. A library-protection idea's whole gain lands in "leak-hit", so L40's "judge by other Δ" and the
  session-grant check (qoder_prompt §10 item 4) score it zero by construction.
- Evidence: `.scratch/libgate_diag/diag.json` `protect.tau0.9_noice.C1_npx`: Δ +0.0054, of which `d_lib_leak` +0.0041.
- Proposal: in the scorecard, split C1 leak-hit into `lib` (`lib_max ≥ LIB_TAU`) and `fp`, and judge C1 ideas by
  `other + lib`.
