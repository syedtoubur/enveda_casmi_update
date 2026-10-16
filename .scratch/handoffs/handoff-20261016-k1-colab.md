# Handoff: K1 (c3-gen-union) on Colab — campaign 20261015-c3-gen-slot2

Written 2026-10-16 for a fresh session. Self-contained; earlier context: `.scratch/handoffs/handoff-20261015-c3-gen-slot2.md`
(campaign decisions), plan `~/.claude/plans/here-i-am-working-crispy-matsumoto.md`, gates K0–K4 in
`research/harness/gate_report.md` (block "2026-10-15: c3-gen-slot2" + its amendment + "K0 MEASURED").

## Goal of this step
Gate **K1**: h = top-1 exact (tautomer-canonical IK14) rate, on the 600 pseudo_c3 molecules, of the union of
(a) the SELFIES decoder per **predicted** formula (beam 64; beam 256 on a subset) and (b) c3v2 mutator edits scored by
decoder teacher-forced log p. **No FPNet** (leaky on pseudo_c3). **DEAD iff h < 3%.** Also: G4.0 (oracle-formula
exact@25, Track-2 kill < 1%), hit@1–4, acyclic vs ring, timsTOF sub-split (all in `build/novel_p4_score.py`).
**Also report the marginal value** (not pre-registered, but it is what matters): union top-1 / top-25 hits on molecules
that are NOT among the 88 whose truth Engine 1's own generator already puts in the BASE list
(`state['base'][mid][1]` of `research/harness/dumps/pseudo_c3/post_dump/state.pkl`, compare via
`casmi_post.canonicalize_smiles_to_ik14`).

## Where things are
- **Code: worktree `E:/casmi/lines/c3-commit`, branch `idea/20261015-c3-gen-slot2`** (pushed to
  github.com/syedtoubur/enveda_casmi, tip `d3714c1`, clean). Work and edit here.
- **Do NOT use `E:/casmi/lines/exploit`**: another agent moved it to `idea/20261015-isdb-cfm-fallback` and has its own
  `casmi_post.py` edits there. It also stashed our c3 WIP as `stash@{0}` (already committed on our branch; leave the
  stash alone). `research/harness/harness.py` defaults `WORKTREE` to the exploit worktree → always pass
  `post=Path("E:/casmi/lines/c3-commit/build/casmi_post.py")` to `load_dump` (`novel_p4_inputs.py` already does, `d3714c1`).
- Main repo `E:/ENVEDA_casmi` branch `specs-scratch-r0` (pushed): gate_report K0, LESSONS L49, idea STATUS files.
- K1 idea folder: `research/exploit/20261015-c3-gen-union/` (STATUS `specced`, gates.md, hypothesis.md). Results go to
  `k1_result.{json,md}` there.
- pseudo_c3 dump: `research/harness/dumps/pseudo_c3/` (post_dump/state.pkl, `eng_lists.json.c3.json`, kernel log;
  gitignored; ident cache warm). Raw kernel outputs also in `E:/casmi/c3_gen/v3_out/`.
- Truths: `E:/casmi/c3cal_check/pseudo_c3.parquet` (= `p1_novel/pseudo_c3.parquet` read by `harness.truths()`).

## Done
- K0 PASS (v3 kernel): no banned truth in PC/ENG/BASE-pool lists; pseudo_c3 E006 MRR@25 0.061, all from Engine 1
  `generate=True` (88 in top 25, 12 at rank 1; leak-inflated because FPNet full1 trained on these spectra).
  L (slot-2 cost): C1_npx 0.058, C1_xlib 0.088, C2 0.048, C2_hard 0.048 → break-even h > ~12% at detector precision
  0.5, > ~5% at 0.7. Observation: BASE#1 is generated on 138/600 pseudo_c3 (55 correct) vs ≈1% of C1/C2 molecules;
  post-scoring demotes 44 of the 55 from rank 1 (candidate detector feature / knob; needs human call, it edits rank 1).
- `E:/casmi/c3_gen/p4_inputs.pkl` built (`python build/novel_p4_inputs.py`, ~5 min): 600 molecules, predicted-formula
  recall 0.932 (`has_formula`, diagnostic only), 445 molecules with mutants, mean 99.2 mutants/molecule.
- `python build/test_novel_p4.py` 5/5; `python build/test_patch_harness_dump.py` 34/34.

## Blocker that the human is fixing (verify first!)
Colab CLI 0.7.4 (WSL, `~/.local/bin/colab`) hardcodes OAuth scope `drive.file` (`colab_cli/auth.py` PUBLIC_SCOPES).
`drivemount` needs full `drive`: tested 2026-10-16 → `ValueError: mount failed`. The human is patching auth.py
(adds `https://www.googleapis.com/auth/drive`, backup `auth.py.orig`) and re-logging in (`colab sessions`, interactive,
in a real WSL terminal — the `!` prompt cannot take the auth code). Auto mode refuses to make that scope change itself.
Note: `colab update` would overwrite the patch.

**Verify:**
1. `python research/bin/colab_run.py --check-login` → must not say "lacks the full Drive scope".
2. Live mount test (≈0.05 CU). From Git Bash, wsl needs `MSYS_NO_PATHCONV=1` and inline `bash -lc '...$VAR...'`
   quoting breaks → use the script: `MSYS_NO_PATHCONV=1 wsl -- bash -l /mnt/e/ENVEDA_casmi/research/queue/_mount_test.sh`
   (new T4 session `mounttest` → drivemount → `research/queue/_mount_probe.py` → stop). Expect `MOUNT_OK` and the probe
   listing `_novel/p0, p1, p3, emb_dbec3a0_lx, p4` and `casmi26-offline-rdkit-2026033` as EXISTS. Or
   `python research/bin/colab_run.py --check-mount`.
   If any Track-2 artifact is MISSING → stop and ask the human.

## Next steps (after the mount works)
1. Upload `E:/casmi/c3_gen/p4_inputs.pkl` to Drive `casmi_2026/_novel/p4/` (find a route: colab CLI has no upload
   command in `--help` (download/ls/exec/edit only) — options: Google Drive desktop if present, a Colab exec that
   receives the file, or the Drive API with the new token; verify the copy by size/hash from a session).
2. Make a runnable script from `build/novel_p4_colab.py`: `colab_run.py` executes a `.py`, and scripts must not call
   `drive.mount()` (replace with `os.path.isdir('/content/drive/MyDrive')` check). `build/mk_novel_p4_colab.py` fills
   `@@TAG@@` source placeholders → add a `--py OUT` option that writes the filled .py; keep the .ipynb path working;
   rerun `python build/mk_novel_p4_colab.py` and `python build/novel_p4_dryrun.py` (CPU dry run incl. resume must pass).
3. Run: `python research/bin/colab_run.py research/exploit/20261015-c3-gen-union <filled.py> --kind val --gate K1
   --fetch <remote raw>:p4_raw.pkl`. Sessions ≤ 3600 s each, resumable from Drive `casmi_2026/_novel/p4/raw/`;
   repeat until done. **Budget:** `cu_cap_val` 4.0 CU ≈ 2 T4-hours (config `research/config.yaml` colab:). The plan's
   estimate assumed A100 (600 × ~2.5 formulas × ~4 s at beam 64 + 150 × beam 256 at ~16 s/mol/formula); T4 is slower.
   Measure s/mol from session 1. If it will not fit: first drop the beam-256 subset (SUB256), then ask the human
   (raise cap / A100 / subset) — never edit the caps yourself.
4. Fetch `p4_raw.pkl` → `E:/casmi/c3_gen/p4_raw.pkl`; `python build/novel_p4_score.py` (in c3-commit) →
   `k1_result.{json,md}` + sidecar `E:/casmi/c3_gen/novel.pkl`; compute the marginal-vs-88 numbers.
5. Record: gate_report "K1 MEASURED" block, idea STATUS, `/update-experiments` if appropriate; commit + push
   (main repo `specs-scratch-r0`; code on `idea/20261015-c3-gen-slot2`). **Checkpoint: human** reads K1 before K2.
6. If K1 PASS (and the marginal is non-trivial): K2 c3-detector per gate_report (features incl. `gen_ice_gap`, "BASE#1
   is generated"), break-even operating point; then K3 knob `NOVEL_SLOT` in `build/casmi_post.py`.

## Gotchas
- Never read raw `.ipynb` JSON (use `.agents/skills/analyze-notebook/scripts/compare.py`; it reports cells only
  coarsely — grep the generated JSON via python for exact strings).
- Never stage .pkl/.npz/.pt/.parquet/.zip. Tracked `build/__pycache__/*.pyc` show as modified after runs → `git checkout`.
- Concurrent agents edit this repo; don't rely on scratch files staying put.
- User prefers subagents for execution (Sonnet) with Opus reviewing; explain results simply.
