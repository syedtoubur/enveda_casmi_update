# Ship note (human checkpoint 2026-10-17)

- Code: `GEN1_KEEP` is merged into `line/c3` (374f4c3, worktree `E:/casmi/lines/c3`). Ship from that worktree.
- Base notebook: `E:/casmi/lines/c3/scratch/baseline/casmi26-v22-rel150-e2r2-clean.ipynb` (V22, LB 0.447–0.451);
  export `scratch/baseline/casmi26_baseline.py`. The probe = V22 + `GEN1_KEEP=True` in its post-scoring stage, nothing else.
- **Paired control:** the config `lines.c3.control` still points at the E006 folder (placeholder). Push an own-account
  **V22 replicate** (the unmodified V22 notebook) as a separate kernel in the same window, with its own control folder
  `research/c3/20261017-control-v22/`, and name it in the submit card (R06). KEEP iff probe ≥ control + 0.004 (gate_report c3-iso block).
- Counters (value-checked, R05): print `gen1_kept N` and `gen1_moved N`. On the visible placeholder test the knob may
  fire 0 times; report it, do not require > 0 on the draft (the hidden re-run is what matters), but the counter line must exist.
- Never `kaggle competitions submit`; the human submits from `research/submit_queue.md`.
