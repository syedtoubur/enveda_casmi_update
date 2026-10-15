# Agent Instructions — Enveda CASMI 2026

## 1. Domain Ground Truth & Scoring
- **Competition**: Predict 2D molecular structures from LC-MS/MS mass spectra.
- **Metric**: `MRR@25` evaluated strictly on tautomer-canonical **InChIKey14** (RDKit 2026.03.3).
- **Invariance**: Stereocenters (R/S, E/Z) and tautomers are ignored. Only 2D graph connectivity counts. Never emit duplicate InChIKey14s in the top 25 (it wastes candidate slots).
- **Evaluation Unit**: Predictions are scored per `molecule_id` (not per spectrum). Aggregate evidence across all spectra (varying collision energies and adducts) for each molecule.
- **timsTOF Precursor Offset**: 100% of test spectra are Bruker timsTOF with a systematic **+1.45 ppm offset** (`CFG.PPM_CENTER = 1.45`). Pre-centering query mass targets sharply reduces decoy candidate dilution.
- **3 Novelty Classes**:
  - **Class 1**: Library match (exact spectral entropy / cosine similarity).
  - **Class 2**: In databases, no spectra (PubChem/COCONUT retrieval + GBDT/neural rankers).
  - **Class 3**: Completely novel (de novo generation via Track 2 or scaffold mutation via Track 1; pure database retrieval has a 0% ceiling).

## 2. How to Navigate This Repo
- **Master Ledger**: Always check [EXPERIMENTS.md](file:///e:/ENVEDA_casmi/EXPERIMENTS.md) to see current SOTA, past scores, deltas, and proven dead ends.
- **Never guess notebooks**: Consult [specs/README.md](file:///e:/ENVEDA_casmi/specs/README.md) to find the canonical files for each track:
  - **Track 1** (SOTA Fusion & Mutator): `specs/track1_retrieval_fusion/`
  - **Track 2** (Novel De Novo SELFIES): `specs/track2_novel_generation/`
  - **Track 3** (MS Simulator): `specs/track3_ms_simulator/`
- **Logging**: When completing a gate or submission, use the `/update-experiments` skill to record results in [EXPERIMENTS.md](file:///e:/ENVEDA_casmi/EXPERIMENTS.md).
- **Pre-Registration**: Write numerical gate thresholds in `gate_report.md` before running measurements on Colab or Kaggle.

## 3. Hard Guardrails & Constraints
- **Kaggle Offline Constraint**: Inference notebooks must run completely offline in ≤ 9 hours on a single T4 GPU. All dependencies (RDKit, SELFIES) must be installed from offline wheels (`--no-index`).
- **Fast-Commit Protocol**: Detect `KAGGLE_IS_COMPETITION_RERUN`. Run a fast smoke test (10–20 molecules) during visible draft commits, but run 100% deep scoring across all test molecules on hidden re-run.
- **Strict Data Boundary**: Never stage or commit `.npz`, `.pt`, `.parquet`, or `.zip` files to Git (enforced by `.gitignore`).
- **No Data Leakage**: In ranker training, never sample queries from structures the FPNet was trained on (the Stage C v1 mistake).
- **Token-Efficient Notebook Inspection**: NEVER read, view, or grep raw `.ipynb` JSON files directly (wastes up to 98% of context on JSON formatting and base64 plots). Always use the `/analyze-notebook` skill or run `python .agents/skills/analyze-notebook/scripts/compare.py [notebooks...]` which converts to `py:percent` and extracts semantic diffs.

## 4. Multi-agent research workflow (`research/`)
- Two lines run in parallel: **exploit** (patches on `casmi-notebook-0-433.ipynb`) and **explore** (`scratch_pkg`). Each line has its own git worktree under `E:/casmi/lines/`, outside OneDrive. Write code only there.
- One idea is one folder, `research/<line>/<idea>/`; its first STATUS line is the state. Roles, the state machine and entry points: `research/README.md`. Config: `research/config.yaml`.
- Antigravity (Gemini Flash): `/cycle <line>` (grill, research, spike, spec), `/critic`, `/daily-digest`. Claude: `/judge`, plus `/run-idea`, which `research/bin/call_claude.py` starts headless (one job at a time).
- Critic and judge checklist: `research/LESSONS.md`. Cite lessons by ID. Only the Reporter edits EXPERIMENTS.md and LESSONS.md.
- Agents push Kaggle kernels (`research/bin/kaggle_push.py`). Only the human submits.

## Agent skills

### Issue tracker

Specs and tickets are local markdown inside `research/<line>/<idea>/`, not GitHub Issues. See `docs/agents/issue-tracker.md`.

### Triage labels

Default vocabulary: needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` + `docs/adr/` at the repo root (created lazily). See `docs/agents/domain.md`.
