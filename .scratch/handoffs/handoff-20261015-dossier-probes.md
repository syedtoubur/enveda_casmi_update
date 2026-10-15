# Handoff: Gemini dossier → LB probes on E006 (2026-10-14 → 2026-10-15)

Self-contained. Read this first, then `CLAUDE.md` and the memory index. Repo `E:\ENVEDA_casmi`; worktrees and data under `E:\casmi`.

## 1. Where we are, in one paragraph
Gemini's research dossier (`research/ideas/enveda_casmi_2026_master_strategy_research_dossier.md`, 12 ideas, "0.440 → 0.491") was fact-checked against the E006 code. It was cut to 7 cheap ideas and each one was run through a pre-registered falsifier. **6 died offline. 2 kernels passed G4 and are `ready-to-submit`: A1 (post-hoc) and B2.** The user submits them next. Nothing from this run has an LB score yet.

## 2. Immediate next steps for the new session
1. **The user submits both** (agents never submit). The kernels are on the pinned image and passed G4:
   - `syedtoubur/casmi26-e006-b2-nl-sim` (v2). Idea folder: `research/exploit/20261014-b2-nl-sim`.
   - `syedtoubur/casmi26-e006-a1-test-network` (v2). Idea folder: `research/exploit/20261014-a1-test-network`.
2. After each LB score: `python research/bin/poll_lb.py --claim`, then the `/update-experiments` skill (EXPERIMENTS.md row). Apply the verdict rule from §3.
3. Stack step, once the scores are in: take the KEEP probes (plus the MAYBEs as a group) and merge their branches into one kernel on E006 v1. Push it, make one confirm submission, and run the G4 runtime check (≤ 9 h T4).
4. Housekeeping for the stack:
   - Add family entries for `NL_LAM`, `NET_VOTE_W` and `FZ_MERGE` to `research/harness/backtests.yaml`. Otherwise `harness/test_suite.py::test_spec_maps_every_knob` fails.
   - Clean up the dossier worktrees (`git worktree remove`) for the dead ideas: exploit-b5, b6, b1, a3, a6.
5. If both A1 and B2 land at ≤ 0.443, the dossier is exhausted. The user then picks the next direction, e.g. the `c3-gen-slot2` campaign (memory) or a new `/campaign-grill`.

## 3. Decisions made in the grill (user, 2026-10-14/15)
- **Gate:** each idea gets a cheap physical falsifier first, then the LB. There is no harness-MRR gate, because the harness is BASE-coupled. `val_clean` (leak-free retrain) is out of scope.
- **Execution model:** Opus writes `spec.md` per idea and verifies. 2–3 **Sonnet subagents** build in parallel, one idea each, each on its own worktree `E:/casmi/lines/exploit-<id>` with branch `dossier/<idea>` off `line/exploit`. Opus reviews and fixes, pushes the kernel, and the user submits. (Memory: `opus-spec-sonnet-build`.)
- **Probes:** one default-off knob per probe on **E006 v1 (LB 0.440)**. **Verdict: LB ≥ 0.444 KEEP; 0.441–0.443 MAYBE (only retried inside the stack); ≤ 0.440 REVERT.** All daily slots may be used, with no replicate, except A1, which is post-hoc and needs a replicate before it joins the stack.
- **Scope:** dropped A2, A5 and B4 (heavy). Dropped A4 (already exists: the C12 DreaMS tail-fill, `dreams_candidates_added=0` everywhere) and B3 (already dead: Track 4b scored 0.399 three times).
- The e180-npool-mutator campaign is paused; its lock was stale.

## 4. Fact-check of the dossier (don't re-derive)
- **B5 premise wrong.** E006 applies no ppm centring anywhere. Fragment tolerance is a fixed `MZ_TOL=0.01` Da (`scratch/e006_embedded/ENG_FILES__pv.py:15`).
- **A6 slots are real.** SLOTS_AGG [2,4,6,8,10] for rel > 600, else GENTLE [4,8,12,16,20]. Promotion fires when (S > 6 and pop ≥ 5) or rel > 250. Code: `casmi26-e006.py:649-670` and `build/casmi_post.py:297-321`.
- **B2:** neutral losses are already used implicitly (FPNet NL embedding, shift-aware library search). The new part is NL similarity against simulated spectra.
- **B6:** per-polarity aggregation already exists; the acid/base complementarity check is new.
- **A1** was never built before. **B1** is new.
- Reading E006 code: use the export `E:/casmi/lines/exploit/scratch/casmi26-e006.py` and the unpacked `scratch/e006_embedded/` (CORE, RUNNER, ENG_FILES__*). Never read raw `.ipynb` files.

## 5. Results per idea
All thresholds were pre-registered in `research/harness/gate_report.md` (blocks dated 2026-10-14 and 2026-10-15) before measuring. Falsifier data:
- timsTOF val: 604 spectra / 245 molecules from `val_scripts/val.parquet`, with `val_key.pkl`;
- BASE decoys: the dumps `research/harness/dumps/*/post_dump/state.pkl`.

| # | Idea | Folder `research/exploit/…` | Result | Key numbers |
|---|---|---|---|---|
| 1 | **B5′** fragment recentring + ppm tolerance | `20261014-b5-frag-ppm` | **KILLED (falsifier)** | AUC truth-vs-decoy 0.571 → 0.573 (10 ppm) / 0.575 (5 ppm), against a +0.02 bar. Drift slope −6 ppm/500 Da, and only 56% of peaks fall within 10 ppm. Tighter tolerance removes true and decoy matches equally. |
| 2 | **A1** test-set network vote | `20261014-a1-test-network` | **K2 KILLED at τ=0.7** (44.4% of edges truthful vs a 50% bar, n=99). **User override: post-hoc probe at τ=0.8, W=0.10** | K1 density 33.5% (τ 0.7) and 19.5% (τ 0.8), measured on the **placeholder** test.parquet (L04). No edges at biogenetic Δm. Replay Δ ≈ 0 (BASE leak). |
| 3 | **B2** NL entropy similarity vs GLACIER, 3rd z-term, NL_LAM 0.5 | `20261014-b2-nl-sim` | **Built, G4 PASS, ready-to-submit** (no falsifier, by user decision) | Commit run: `B2 runner patched`, `nl_scored 754`, `nl_rerank_mols 11`. |
| 4 | **B6** dual-polarity acid/base demotion | `20261014-b6-dual-polarity` | **KILLED** | v1 SMARTS demotes 23.7% of truths vs 22.2% of decoys, so it doesn't discriminate. Loosening only reaches 2.0% (still > 1%) while decoy hits drop to 3.4%. |
| 5 | **B1** even-electron rule penalty | `20261014-b1-eer` | **KILLED** | AUC 0.490 (chance). At λ=0.4 it moved one truth, downward. |
| 6 | **A3** formula explained-intensity rerank | `20261014-a3-formula-eif` | **KILLED on K1** | True formula is EIF-rank-1 in only 56.5% of cases (85% bar). K2 is void: it was computed on the fast-commit placeholder (388/400 rows `CCO`). |
| 7 | **A6** f·z common-scale merge (`FZ_MERGE`) | `20261015-a6-fz-merge` | **KILLED (harness gate)** | C2 "other" Δ −0.036/−0.013/−0.001 at m = 0/50/150. Rank-1 displacements of 23–40 per 400. C1_npx CI upper bound ≤ −0.005. The only +Δ (C2 m=150, +0.012) is entirely leak-hit. Details in `verdict.md` and `gate/`. |

**Pattern learned.** At E006's 0.01 Da tolerance, every fragment-explanation physics check (B5′, B1, A3) fails to separate truths from same-mass BASE decoys. Merge and promotion changes (A6, like promo-soften before it) read only as BASE-coupled leak effects.

## 6. Built artefacts (branches are local; nothing is pushed to GitHub)
- **A1:** branch `dossier/a1-test-network` (96a3143, da00679).
  - Worktree: `E:/casmi/lines/exploit-a1`.
  - Code: `build/casmi_post.py` adds `modified_cosine`, `test_network`, `net_vote_rows` and knob `NET_VOTE_W`.
  - Patcher and test: `build/patch_a1_test_network.py`, `build/test_patch_a1_test_network.py`.
  - The ship cell is the last cell, after `FACTORIAL_REPORT`. Ranks 1 and 11–25 are frozen.
  - Commit run: `NET_VOTE {"net_mols": 0 …}`, which is expected with only 12 molecules. Check the hidden-rerun density in the full log.
- **B2:** branch `dossier/b2-nl-sim` (5d0666c).
  - Worktree: `E:/casmi/lines/exploit-b2`.
  - The patcher `build/patch_b2_nl_sim.py` copies the GLACIER package to `/kaggle/working/gl_pkg_nl` and text-patches `gl_runner.py`, which writes `gl_out.json.nl.json`. It changes 4 notebook cells: cell 7, cell 9, the E004 overlay and `rerank_row`.
  - Knob `NL_LAM` in casmi_post.
  - The GLACIER source was downloaded to `E:/casmi/data/sim_pkgs/glacier/` (weights deleted).
- **A6 (dead):** branch `dossier/a6-fz-merge` (49a9a07). Its harness dumps are installed as `research/harness/dumps/C2_a6` and `C1_npx_a6`:
  - new runner with `pool_fz` and `pc_rawfz`;
  - ICE grafted from the old C2 and C1_npx dumps, because ICE failed on the pinned cp312 image.

  These dumps are the only ones with per-candidate pool f·z, so they are reusable for any future merge idea.

## 7. Operational gotchas hit this session (each cost time)
- **Always push through `research/bin/kaggle_push.py`.** It pins the control image (`docker_image` `sha256:37c64f7…`, Python 3.12). A1 and B2 v1 were pushed another way, landed on Py 3.13, and died at E006's `assert python == 3.12` (cell 7, 46 s).
- **The pinned cp312 image breaks the 0.433 harness notebook's ICE preseed** (its RDKit wheel is cp313). Every harness dump on it ends with `ICE FAILED`. Graft `ice` from an older dump of the same cohort (`state['ice'] = old['ice']`); the paired comparison stays valid.
- **Kaggle allows max 2 concurrent GPU sessions.** Pushes fail with "Maximum batch GPU session count of 2".
- **A failed push leaves `kaggle_push/counters.txt` behind**, and preflight R05 then reads that stale file. Delete `<idea>/kaggle_push/` and set STATUS back to `built` before retrying.
- **Preflight rules:**
  - R01: any `test.parquet` mention must say "placeholder" or L04.
  - R04: cited files must use full paths, e.g. `E:/casmi/lines/exploit/scratch/e006_embedded/…`.
  - R05: at least one counter must be value-checked, not only `?`-prefixed. A counter on a constant (e.g. `"tau": `) works for commit runs.
- **The local `test.parquet` and the saved commit outputs are placeholders** (12 molecules scored, the rest `CCO`). Never compute reach from them.
- **Kaggle OAuth expired mid-session.** Calls fail with "Permission 'kernels.get' was denied". The fix is `! kaggle auth login --force`.
- **Harness:** `harness.py compare` has no leak-hit/other split. Call `harness.cohort_score(shard, A, B, post)` directly.
  - The E006 base cfg is `METRIC_FUSION=true,ICE_FALLBACK=true,FUSION_WINDOW=40`.
  - Pass `--post` pointing at the idea worktree's `build/casmi_post.py`.
- **Background shell loops can be killed** on low memory. Don't rely on them across idle periods.

## 8. Key files
- Plan: `C:/Users/h/.claude/plans/pasted-content-id-ff53-giving-our-sunny-scroll.md`
- Dossier: `research/ideas/enveda_casmi_2026_master_strategy_research_dossier.md`; earlier candidates in `research/ideas/20261014-candidates.md`.
- Per-idea: `research/exploit/2026101{4,5}-*/` (`spec.md`, `falsifier.md`, `build_report.md`, `verdict.md`, `STATUS`, `gates.md`, `kaggle_output/`).
- Pre-registration: `research/harness/gate_report.md` (end of the file).
