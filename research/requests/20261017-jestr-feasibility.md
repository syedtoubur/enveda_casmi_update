# Feasibility request — contrastive spectrum↔molecule channel (idea 2)

status: resolved   # licence/weights questions answered 2026-10-17 (Qoder web research)
asked_by: /ideate 2026-10-17 (jestr-contrast-channel)

## Question that was open
Can a contrastive spectrum↔candidate joint-embedding **direct retrieval** scorer be used
legally and run offline (Kaggle cp313, one T4, ≤ 9 h, `--no-index`)? Reusing an existing
open model beats training one (ideate step 3).

## Findings (per model)

### JESTR — HassounLab/JESTR1 (Kalia 2025, Bioinformatics, 24 cites) — **GREEN**
- **Licence: MIT.** ✓ competition-legal (freely + publicly available, §A8).
- **Weights: RELEASED** for **NPLIB1** and **MassSpecGym** (GPU/T4-compatible; trained A100
  CUDA 11.8 but inferable on T4). ✓ no training needed to prototype.
- **Mechanism:** contrastive joint embedding of (spectrum, molecule graph); rank candidates
  by **cosine similarity** — direct retrieval, NOT predict-then-match. This is the orthogonal
  evidence type the pipeline lacks.
- **Molecule side:** encodes arbitrary candidate SMILES (builds DGL graph per InChIKey on
  the fly when `molgraph_dict` absent) → can score BASE∪ENG∪PC union candidates.
- **Dependencies:** PyTorch + **DGL**. Precedent: this repo already ships a `shim/dgl`
  stand-in for GLACIER → offline cp313 packaging has a template, medium risk.
- **Leak:** NPLIB1/MassSpecGym are GNPS/riken/MONA-class → overlap val queries (L19). MUST
  read on scaffold-banned **C2_hard**, **pseudo_c3**, and the honest cohort (idea 1), never
  on leak-hit C2.
- **Correlated-evidence risk (L11):** cosine vs `f·z` may be ~near-copies. Falsifier is
  mandatory: Spearman(JESTR-rank, `f·z`-rank) > 0.85 → dead; Δ_other ≤ +0.003 → dead (L45).

### MIST / MIST-CF — samgoldman97/mist — **AMBER**
- Contrastive embed + retrieval quickstarts exist; **CSI2022 cannot be redistributed**,
  retrieval HDF > 5 GB; augmentation via an ms-pred forward model. Heavier to stage offline
  than JESTR; licence not confirmed MIT in the README (re-check LICENSE file if chosen).

### DreaMS — pluskal-lab/DreaMS — **GREEN but partial**
- Open, already vendored, byte-identical on Kaggle T4 (G0 PASS). It is a **spectrum**
  encoder; there is no shipped turnkey *molecule* tower for cosine retrieval, so it serves
  as one half of a custom contrastive head, not a ready scorer.

### GLMR (arxiv 2511.06259) / MS-BART (NeurIPS 2025) — **RED this cycle**
- No public repo/weights found → not offline-stageable now. Recheck after code release.

## Conclusion
**Use JESTR pretrained MassSpecGym weights** as the concrete realization of
`jestr-contrast-channel`. No model training required to test it. Cheapest correct next
experiment = a **CPU/GPU-light inference spike** that (a) stages JESTR + the DGL shim,
(b) scores the union candidates already in a dump, (c) computes Spearman(JESTR, f·z) and
ΔMRR_other on C2_hard / pseudo_c3. This is the pre-slot falsifier for idea 2 and needs no
Kaggle submission.

## Open for the human
Approve building the JESTR inference spike (idea 2 falsifier) as the next loop stage? It is
GPU-light (single T4 inference over dumped candidates) and does NOT require the honest
cohort or the per-spectrum dump re-run to produce the go/no-go correlation number.
