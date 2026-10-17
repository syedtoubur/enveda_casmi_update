# Submit card: gen-rank1-keep (R1, GEN1_KEEP) — campaign 20261017-c3-iso

- **Kernel:** `syedtoubur/casmi26-gen-rank1-keep` (probe), from `E:/casmi/lines/c3/casmi26-v22-gen1-keep.ipynb`
  (`build/patch_gen1_keep.py`, branch `idea/20261017-gen-rank1-keep-ship` 3846918; knob in `build/casmi_post.py` 374f4c3).
- **Change:** V22 + one last step. If Engine 1's generated structure (formula tagged `|gen`) is BASE #1 and its ICEBERG
  cosine is >= every other BASE top-5 candidate's (at least one comparison exists), it is pinned at rank 1 of the V22
  row (deduped by metric identity, cut at 25). It is applied to the finished V22 `submission.csv`, so only the rows
  where it fires change. No FPNet value is read by the trigger.
- **Scorecard (pseudo_c3, harness_result.json, QUEUE):** pseudo_c3 MRR 0.05619 -> 0.06619, delta +0.0100
  [+0.0045, +0.0163], 10 up / 0 down, leak-free "other" delta 0.0, 0 rank-1 displaced.
  C2 +0.0029 [0.0, +0.0088] (2 up); C1_npx, C1_xlib, C2_hard, C2_np 0.0. 37 fires on 600 pseudo_c3 molecules.
- **Flags:** (1) pseudo_c3 gain sits on leak-hit molecules (delta_leak_hit +0.107 on 56, other 0.0): ASSUMED prior (L40),
  and on the hidden test FPNet has not seen the spectra, so generated #1 precision is lower (attack A5, LB decides).
  (2) Attack A2 ran a proxy (raw C1/C2 fires and rank effects) accepted by the human. (3) On the 12-molecule draft the
  knob may fire 0 times: `gen1_kept`/`gen1_moved` are `?` counters there; `gen1_checked` (> 0) proves the stage ran.
  The draft is compared with the E006 control draft (config placeholder), so `g4_min_changed: 1` is met by V22-vs-E006
  differences, not by the knob firing; read the hidden-rerun counters for that.
- **Counters (value-checked):** `GEN1_KEEP gen1_checked N` (> 0), `?GEN1_KEEP gen1_kept N`, `?GEN1_KEEP gen1_moved N`.
- **Paired control:** `syedtoubur/casmi26-control-v22-replicate` (folder `research/c3/20261017-control-v22`), an
  unmodified own-account V22 replicate pushed in the same window and on the same image.
- **KEEP is decided by poll_lb.decide()** (probe >= paired control + 0.004, counters fired on the hidden rerun,
  replicate before it becomes the base). The human submits.
