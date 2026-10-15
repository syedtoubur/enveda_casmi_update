# Handoff: campaign 20261015-c3-gen-slot2 (Class 3 on top of E006 0.440)

Plan: `~/.claude/plans/here-i-am-working-crispy-matsumoto.md`. Charter: `research/campaigns/20261015-c3-gen-slot2/charter.md`.
Gates K0–K4 + the leak-free amendment: `research/harness/gate_report.md`, last block (2026-10-15).

## Decisions (user, 2026-10-15)
- Gated insertion in the top 4: ONE generated candidate at slot 2 on detector-fired molecules (never rank 1).
- Generators: Track 2 SELFIES decoder (per predicted formula) ∪ c3v2 mutator (analog edits).
- **No FPNet in the novel channel**: pseudo_c3 queries are train.parquet rows FPNet full1 trained on. Scale = decoder
  masked log p (`teacher_logp` for edits == beam score on a leaf, tested); ICEBERG cosine later; `gen_ice_gap`.
- Run as a campaign; GPU approved for the pseudo_c3 dump and Track 2 P4.

## Done
- Old campaign `20261014-e180-npool-mutator` stopped (state.json stop + log), stale lock removed.
- `research/harness/harness.py::truths()` adds pseudo_c3 truths (cohort pseudo_c3, cls 3) from `p1_novel/pseudo_c3.parquet`.
- Exploit worktree, branch `idea/20261015-c3-gen-slot2` (uncommitted; branched from idea/20261014-e180-drop with its
  pending lib-isomer edits):
  - `build/patch_harness_dump.py --shard pseudo_c3`: query rows = cohort train_rows; library ban plain IK14 ∪
    score_key (+ tautomer rows near a truth mass); Engine 1 `P.window` ban (generated pid −1 allowed); Engine 2
    `build_pool` ban via the .pth hook (`HX_BAN_KEYS`); PubChem probe pool_keys + runner best_pool_fz ban; audit cell;
    **Engine 2 analog export** `eng_lists.json.c3.json` (mutator anchors). Tests: `build/test_patch_harness_dump.py` 31/31.
  - `build/novel_p4.py` (+ `test_novel_p4.py` 5/5): predicted formulas, `teacher_logp`, union, K1 report.
  - `build/novel_p4_inputs.py` (local CPU): dump → `E:/casmi/c3_gen/p4_inputs.pkl`.
  - `build/novel_p4_colab.py` → `casmi_novel_p4_colab.ipynb` (`build/mk_novel_p4_colab.py`); CPU dry run
    `build/novel_p4_dryrun.py` passes (incl. resume).
  - `build/novel_p4_score.py`: K1 verdict → `research/exploit/20261015-c3-gen-union/k1_result.{json,md}` + sidecar
    `E:/casmi/c3_gen/novel.pkl`.
- Catalog: `syedtoubur/casmi26-c3cal-cohorts` (pseudo_c3.parquet, content-identical to local) registered.
- Kaggle: `syedtoubur/casmi-0433-harness-pseudoc3` **v2** pushed 2026-10-15 (v1 lacked the analog export; it may still
  run, cancel it in the Kaggle UI if so). Watcher heartbeat: `research/exploit/20261014-pseudo-c3-dump/kaggle_watch/`.
- Runtime fact: a 400-molecule val dump runs ~3.0 h end to end → E006 hidden headroom ≈ 5.5 h.
- K4 risk noted: masked beam 256 ≈ 16 s/mol/formula on A100; production must use beam ≤ 64 and/or fired molecules only.

## v2 failed at the ban audit (2026-10-15) → v3 pushed
- v2 ran 8,367 s then died at the audit: `ENG lists 8` (C3_00051, 00058, 00151, 00277, 00419, 00487, 00556, 00581).
  Cause: Engine 2 pool keyed by plain IK14, lists re-keyed by `canon_key` (tautomer-canonical) → truth tautomers
  under other plain keys leaked (`engine2 pool removed 1 of 774327`). Engine 1 clean (pool key == score_key, 300/300).
  No post_dump was written. Lesson L49.
- Fix (`build/patch_harness_dump.py`, tests 34/34): the Engine 2 hook runs `canon_key` on pool rows within 10 ppm of
  a truth mass (`hx_ban.json` now carries `masses`); `HX_PID[mid]` = Engine 1 key→pid graft in the C7 cell;
  the audit fails on any banned key in BASE without pid −1 proof. All 8 v2 leaks sit at 0 ppm from a truth mass.
- **v3 pushed** 2026-10-15 (~2.5 h + ICE). Watcher: `research/exploit/20261014-pseudo-c3-dump/kaggle_watch/`.

## Next (in order)
1. When v3 finishes: `kaggle kernels output syedtoubur/casmi-0433-harness-pseudoc3 -p <tmp>`; install
   `post_dump/` as `research/harness/dumps/pseudo_c3/post_dump/` and `eng_lists.json.c3.json` next to it. Check the log
   counters: `C3 ban keys`, `C3 library ban rows`, `HARNESS c3 ban engine1 pool pids`, `engine2 pool removed N of …
   (plain …, tautomer T)` with T ≥ 8, `HARNESS c3 analog export`,
   `HARNESS c3 audit: … generated (pid -1, allowed) G, pool 0; PC lists 0, ENG lists 0`, `POST EQUIV ok 600 of 600`.
   G is the number of truths Engine 1's generator re-derived exactly (v2 showed 88 unverified); note it for K1.
2. K0: `harness.py warm --dump pseudo_c3` then BASE MRR on pseudo_c3 under the E006 cfg; P(r) on C1/C2 "other" → L.
   Record in gate_report + `research/exploit/20261014-pseudo-c3-dump/verdict.md`; STATUS instrument-ready.
3. `python build/novel_p4_inputs.py` → upload `E:/casmi/c3_gen/p4_inputs.pkl` to Drive `casmi_2026/_novel/p4/`;
   run `casmi_novel_p4_colab.ipynb` on Colab GPU (A100 ≈ 600 × ~2.5 formulas × ~4 s at beam 64 + 150 × beam 256);
   download `p4_raw.pkl` to `E:/casmi/c3_gen/`; `python build/novel_p4_score.py` → K1. **Checkpoint: human.**
4. If K1 PASS: c3-detector (K2) on dumps C1_npx, C1_xlib, C2, C2_hard (other) vs pseudo_c3, features incl. gen_ice_gap
   (needs ICEBERG on the union top-N), break-even operating point. Then c3-slot2-route knob `NOVEL_SLOT` in
   `build/casmi_post.py` + rule fixture (never edits rank 1) → K3 replay → K4 → LB with a paired E006 replicate.
