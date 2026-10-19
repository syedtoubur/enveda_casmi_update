# Session Handoff Dossier: Antigravity Desktop Autonomous Loop Bootstrap

**Generated:** 2026-10-17 07:36:00 UTC
**Context Size:** ~154k tokens (Smart Zone Ceiling Reached)
**Target Line:** Track 2 — Exploit Loop in `E:/casmi/lines/antigravity-exploit` (Parallel to Claude Code Track 1 on K2 detector)

---

## 1. High-Level Mission & Active Objective

Establish a parallel, dual-track autonomous research architecture for the **Enveda CASMI 2026** competition:
- **Track 1 (Claude Code)**: Dedicated exclusively to Campaign `20261015-c3-gen-slot2` (the K2 Class-3 detector in `E:/casmi/lines/c3-commit`, branch `idea/20261015-c3-gen-slot2`).
- **Track 2 (Antigravity Desktop App)**: An independent autonomous research exploit loop powered by Antigravity Pro. It pulls the new SOTA baseline kernel `easonyanyan/casmi26-v22-rel150-e2r2-clean` (LB 0.447–0.451), synchronizes the offline replay harness (`build/casmi_post.py`), and executes an 8-idea prioritized portfolio derived from `research/ideas/innovative_ideas.md` and `.scratch/CORPUS.md` targeting **0.470+ MRR@25**.

---

## 2. Settled Decisions (Do NOT Re-Litigate)

- **Dedicated Worktree Isolation**:
  - *Decision:* Antigravity operates exclusively inside a new, isolated worktree `E:/casmi/lines/antigravity-exploit` on branch `line/antigravity-exploit`.
  - *Rationale:* Completely prevents git lock collisions, branch overwrites, or stash conflicts with Claude's active K2 run in `E:/casmi/lines/c3-commit` or dirty exploit lines.
  - *Impact:* Antigravity can execute full git lifecycles (`idea/<slug>` branches, test suites, commits) without touching Claude's files.
- **Baseline Kernel Definition**:
  - *Decision:* The operational baseline is `easonyanyan/casmi26-v22-rel150-e2r2-clean` (LB 0.447+), pulled via `kaggle kernels pull`.
  - *Rationale:* Rather than manually rebuilding the Apex SOTA components from scratch on E006, this kernel already incorporates tuned PubChem promotion (`rel > 150.0`), metric-keyed E2R2 rank-2 insertion with canonical InChIKey14 deduplication, and clean database curation.
  - *Impact:* Antigravity starts research directly on top of the 0.447+ baseline.
- **Offline Harness Replay Parity (Cycle 0 Gate)**:
  - *Decision:* Antigravity converts the baseline notebook to Jupytext percent format, updates `build/casmi_post.py` in its worktree to match the post-processing logic, and runs a zero-change calibration scorecard on the multi-cohort battery.
  - *Rationale:* Guarantees fast, honest, 15-second offline evaluation on 1,390 molecules before spending any Kaggle GPU budget.
- **Iterative Refinement vs. Blanket Bans (The CFM-ID 4.0 Learning Principle)**:
  - *Decision:* Do not discard physical signals or algorithms outright simply because an earlier implementation failed. CFM-ID 4.0 (Idea A2) dropped LB from 0.440 to 0.435 because it was plugged in as a rigid hierarchical fallback gate (`Glacier → ICBM → Iceberg → CFM-ID`). In the new portfolio, CFM-ID is re-architected into an independent soft RRF rank-fusion channel (+0.35*z(cfm)) or confidence-calibrated margin covering [M-H]- negative mode where GLACIER is blind.
  - *Impact:* Evaluated empirically on the multi-cohort battery (`C2_np` negative-mode queries) rather than killed a priori.
- **Multi-Cohort Evaluation & `pseudo_c3` Role**:
  - *Decision:* Gating is evaluated on non-advisory cohorts `C1_npx`, `C1_xlib`, `C2`, `C2_hard` strictly on `other` (non-leak) subsets. The 600-molecule `pseudo_c3` cohort is included in every scorecard as an advisory diagnostic monitor for off-base reach and novel candidate recall.
- **Kaggle Kernel Push & Submit Autonomy**:
  - *Decision:* Agent runs `kaggle_push.py` for detached G4 execution verification. Submissions are strictly human-only by default; session-scoped autonomy can be granted temporarily via the exact chat token `SUBMIT AUTONOMY GRANTED FOR THIS SESSION`.

---

## 3. Modified & Created Files

- [`e:\ENVEDA_casmi\.scratch\antigravity_prompt.md`](file:///e:/ENVEDA_casmi/.scratch/antigravity_prompt.md) - Turnkey bootstrap instruction file for the Antigravity Desktop app, specifying the autonomous research state machine, runtime probe, portfolio of 8 candidate directions, offline gating criteria, and dual-agent isolation rules.
- [`e:\ENVEDA_casmi\.scratch\CORPUS.md`](file:///e:/ENVEDA_casmi/.scratch/CORPUS.md) - Master research record updated with Section A2.1 (2026-10-17 SOTA Addendum) capturing the new 0.447+ baseline, dual-track operations, and root-cause diagnosis of prior CFM-ID integration.
- [`e:\ENVEDA_casmi\.scratch\handoffs\latest.md`](file:///e:/ENVEDA_casmi/.scratch/handoffs/latest.md) - Active handoff dossier.
- [`e:\ENVEDA_casmi\.scratch\handoffs\handoff-20261017-antigravity-bootstrap.md`](file:///e:/ENVEDA_casmi/.scratch/handoffs/handoff-20261017-antigravity-bootstrap.md) - Versioned archive snapshot of this handoff.

---

## 4. Current Work in Flight

- Both [.scratch/antigravity_prompt.md](file:///e:/ENVEDA_casmi/.scratch/antigravity_prompt.md) and [.scratch/CORPUS.md](file:///e:/ENVEDA_casmi/.scratch/CORPUS.md) are fully aligned, verified, and saved to disk.
- Claude Code is actively executing in `E:/casmi/lines/c3-commit`.
- Antigravity Desktop is ready to be launched with the newly minted bootstrap prompt.

---

## 5. The Immediate Frontier (Next Steps)

1. **Launch Antigravity Desktop Session**:
   - Open a fresh chat in the Antigravity Desktop App and supply the bootstrap trigger:
     ```text
     Please read .scratch/antigravity_prompt.md and execute the autonomous research loop starting with §1 (runtime probe) and §3 (Cycle 0: baseline pull, worktree initialization, and calibration scorecard).
     ```
2. **Execute Cycle 0**:
   - Create worktree `E:/casmi/lines/antigravity-exploit`.
   - Run `kaggle kernels pull easonyanyan/casmi26-v22-rel150-e2r2-clean -p E:/casmi/lines/antigravity-exploit/scratch/baseline -m`.
   - Convert to `casmi26_baseline.py` via Jupytext.
   - Verify `build/casmi_post.py` matches baseline post-processing, run pytest, and execute `harness.py scorecard` to establish baseline scores on `C1_npx`, `C1_xlib`, `C2`, `C2_hard`, and `pseudo_c3`.
3. **Execute Cycle 1 (`e180-drop-dreamsnp`)**:
   - Implement `E180_DROP=True` + fix `from pathlib import Path` for DreamsNP tail backfill in Cell C12.
   - Run harness scorecard, verify $\Delta_{\text{other}} > 0$, run gate evaluation, and push G4 kernel.
4. **Advance through Prioritized Portfolio**:
   - Idea 2: `tautomer-canon-unify` (Idea A6)
   - Idea 3: `cross-formula-rerank` (Idea B1)
   - Idea 4: `bpi-adaptive-snr` (Idea B5)
   - Idea 5: `cfmid-isdb-rrf` (Idea A2 Refined)
   - Idea 6: `v4b-union-xenc` (Idea A1)
   - Idea 7: `sub5mda-nl-rda` (Idea B7)
   - Idea 8: `conformal-c3-router` (Idea A7)

---

## 6. Recommended Skills & Boot Command

- Recommended skills for downstream execution: `/executing-plans`, `/test-driven-development`, `/analyze-notebook`.

### Copy-Paste Boot Prompt for New Session:
> "Resume work from [latest.md](file:///e:/ENVEDA_casmi/.scratch/handoffs/latest.md). Read the Settled Decisions and immediately proceed with Step 1 of the Immediate Frontier."
