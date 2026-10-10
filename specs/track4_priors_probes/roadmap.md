# Track 4 — Priors & Probes: a fast-submission track on top of the 0.405 notebook

## Context

`evenda-casmi-inferencesupernext.ipynb` scores **0.405**, the best so far. A semantic diff against the 0.399 parent (`casmi26-v4n-engine-fusion-union-lb-0-399.ipynb`) shows that only two changes in it are live:

1. **A PubChem popularity prior.** `casmi26-pubchem-popularity-prior` (a public dataset) adds `pool_lsid.npy` and `pool_lpmid.npy`. Cell C7 re-sorts the v4n BASE pool candidates by `z(scs) + 0.15·pop`, and this happens **after** the `TOPN=60` cut.
2. **GLACIER restricted to `[M+H]+` spectra** in C8. A `'|gen'` suffix also isolates generated candidates' formula groups, which is a side-effect fix for the mutator's "friendly fire" problem.

That makes popularity a **new orthogonal axis**: it is the first spectrum-blind signal in the lineage. The literature agrees that it is strong:

| Source | Result |
|---|---|
| CASMI 2016 | "database boosting" took top-1 from 28% to 70–93% |
| MetFrag | reference/patent counts lifted top-1 from 58% to 70% |
| MassSpecGym v1.5 audit | ranking by PubChem order alone gives about 49% Hit@1 |

Right now the prior is applied in only **one** of the four lists (v4n BASE), and only to re-order candidates that are already there. The engine-A list, the PubChem-only list, the fused list and the pre-cut pool never see it.

**Goal.** Get as many LB probes as possible (2–3 per day), each one a single additive change on top of the current best. Order them cheapest first, and move to the next wave as soon as a wave stops paying.

**Facts that constrain the plan:**
- The adduct is given per spectrum.
- There is no CCS, ion-mobility or RT data in the test set, so CCS and RT levers are dropped.
- Compute available: Colab A100, Kaggle T4, and the local CPU.
- New Kaggle datasets built on Colab are approved.

## Step 0 — Housekeeping (first, about 30 min)

- Write **`updated_knowledge_base.txt`** at the repo root. It should contain:
  - the full literature knowledge base from the research subagent (sections A–F, the top-10 lever list, and the sources);
  - the supernext pipeline audit and diff from the notebook subagent.
- Create `specs/track4_priors_probes/` with `roadmap.md` (this plan) and `gate_report.md` (pre-registered keep/kill rules, below). Add the Track 4 row to `specs/README.md`.
- Use `/update-experiments` to log supernext as **row 24, 0.405 (+0.006)**, and update the "Current SOTA" line, which still reads 0.399.

## Notebook-generation rule

Never hand-edit `.ipynb` files. Add `build/patch_supernext.py` on the pattern of `build/patch_v4n_mutator.py`. It reads `evenda-casmi-inferencesupernext.py`/`.ipynb` and emits one variant notebook per flag (`--pop-precut`, `--pop-fused`, `--c1-freeze`, `--pc-pop`, …), so each probe is a single, reviewable delta. Every flag must sit inside the existing `try/except` fallback (rule R5 in `plan_v2.md`) and honour `IS_RERUN`.

## Pre-registered decision rules (copy into `gate_report.md`)

- Each probe contains exactly one change on top of the current best.
- **Keep** a change if LB ≥ best + 0.001. Ties go to the simpler variant. Revert anything that loses.
- After two consecutive flat or negative probes within a wave, move to the next wave.
- Every rerun log must print the counters: `top-1 changed in N`, list overlap, and `pop stats` (min, median, p99 of the prior term). If a probe moves the LB but its counters show no change, treat the move as noise.
- Never weight a new signal at 1.0 on its first appearance (R4). For the forward models, keep λ ≤ 1.0.

## Wave A — free probes (days 1–2). Reuse the attached datasets; no new data needed.

| # | Change (in the supernext cell) | Why | Expected ΔLB |
|---|---|---|---|
| A1 | **Apply the prior before the `TOPN` cut** (C7): compute `f = z(scs) + μ·pop` over *all* `V.run` candidates, then cut to 60. This also implements the dead `POP_UNION`. | Truths sitting at ranker rank 61+ with high popularity become reachable, so this is coverage the current re-order can't provide. | +0.002…+0.006 |
| A2 | **Apply the prior to the fused list** (C10): after `fuse2`, add `μ_f·z(pop)` for keys found in the pool's IK14→pop map, before the ICE/GL rerank. Start at μ_f = 0.15. | Engine-A and PubChem-slot candidates currently get no prior. | +0.001…+0.004 |
| A3 | **Two-point μ sweep** on whichever of A1/A2 wins: try μ ∈ {0.08, 0.30}. First log the scale of `lsid+lpmid` (a commit run is enough for the scale). | 0.15 was a single guess, and the scale of the term is unknown. | ±0.003 |
| A4 | **Class-1 top-1 freeze**: if `lib_max ≥ 0.9`, pin BASE rank 1 through C8/C10. | Popularity and forward reranks can currently demote confident library hits; the λ=2 run lost 0.034 this way. | 0…+0.003 |
| A5 *(optional, attribution only)* | `POOLPOP_MU=0` with `'|gen'` kept | Separates the two supernext effects. Run it only if a probe slot would otherwise go unused. | — |

## Wave B — prior v2 dataset (Colab build on days 1–3, probes on days 3–5)

**Build `build/prior_v2_colab.ipynb`.** It downloads the PubChem FTP `CID-PMID`, `CID-Patent` and `CID-SID` files, PubChemLite, and LOTUS (structure–organism pairs). It collapses every source to tautomer-canonical IK14, using the same `score_key` as the notebook, and computes the following per key:

- `log1p(pmid)`
- `log1p(patent)`
- `log1p(sid)`
- `in_pubchemlite`
- `lotus_plant_occurrences`
- `n_plant_families`
- `in_coconut`
- NP-likeness

Coverage must include:
- the v2 pool (`pool_meta` order);
- the **PubChem tier** (`pc_smiles` / `pc_off`);
- engine A's COCONUT + BIO.

Upload the result as the private dataset `casmi26-prior-v2` (`keys.npy` plus feature arrays), with a MANIFEST containing an assertion that `len == len(pool_meta)`.

| # | Change | Why | Expected ΔLB |
|---|---|---|---|
| B1 | **Popularity-ordered PubChem-only channel** (C4/C9): order `pc` by `z(pc_fz) + μ·pop` instead of f·z alone. | Class 2 is "in PubChem, no spectra", which is exactly where popularity predicts best. Today this list is ordered by f·z only and is capped at 5 slots. | +0.003…+0.010 (best single bet) |
| B2 | Increase the PubChem slots to 6–8 for low-`rel` molecules **only if B1 pays**. | More slots are only worth having once the slots are better ordered. | +0…+0.003 |
| B3 | Replace `lsid+lpmid` with the richer prior: `pmid + patent + lotus_plant + in_pubchemlite`, combined as a fixed sum (no fitting). | The test molecules are plant natural products, and LOTUS plant occurrence is the NP-specific version of popularity. | +0.001…+0.004 |

**Not doing:** a learned "measured-ness" classifier. Class-2 truths by definition **have no public spectra**, so a model trained with train-set structures as positives learns "has spectra" and would penalise exactly the class-2 truths (the Skinnider confound). Fixed literature/occurrence counts don't have that bias.

## Wave C — novel-structure channel, run in parallel on Colab (weeks 1–2). Gated before any LB probe.

DreaMS+SELFIES reached 8.17% exact@25 on `pseudo_c3`. The research points to a stronger and cheaper replacement: **MS-GPT** (Apache-2.0, github.com/VIKI623/MS-GPT). It samples a molecule LM using a band of thresholded fingerprints drawn from a predicted fingerprint posterior, and reports 23.9% top-1 on MassSpecGym and 29.8% on NPLIB1. Generation is autoregressive and cheap.

1. **C0 feasibility (Colab):**
   - Run MS-GPT with its own encoder on `p1_novel/pseudo_c3.parquet`, reusing the eval helpers from the Track 2 P3 notebooks.
   - **Gate:** exact@25 ≥ 12% (it must clearly beat the 8.17% DreaMS arm) at ≤ 20 s per molecule on a T4.
   - **Kill:** if the gate fails, stop Wave C.
   - **Check:** confirm the encoder's training split; it must use MassSpecGym-only weights (`casmi-simulator-checkpoint-licence`).
2. **C1 integration (only if C0 passes):**
   - Run MS-GPT as a subprocess that writes `{mid: {smiles, keys}}`.
   - Fire it only on molecules flagged by `build/c3_gate.py`.
   - **Route, don't fuse** (lesson from `c3_routing.md`): splice the candidates into reserved ranks, e.g. 2, 5, 9, through `merge()`.
   - Formula groups get the `'|gen'` suffix, so the candidates can't swap with a pool truth.
   - Run ICE on the generated candidates (rule R2).
   - Pre-register GM-style displacement gates on `p1_novel/c2_displace.parquet`.

## Deferred, with reasons

| Lever | Why it's deferred |
|---|---|
| NAP network consensus | Largely duplicates engine A's analog propagation (`N_ANALOG=200`), so it isn't orthogonal. |
| Calibrated or LambdaMART fusion | There is no validation set aligned with the LB, and tuning to local val cost 0.015 before. |
| Contrastive retrieval (FLARE/JESTR) | Weeks of training work. It is the next orthogonal source **after** Wave C. |
| CCS / RT | Not present in the test set. |

## Critical files

| File | Role |
|---|---|
| `evenda-casmi-inferencesupernext.ipynb` / `.py` | Base for every probe. Cells C1 (config), C4 (PubChem), C7 (BASE + popularity), C9 (merge), C10 (`fuse2`). |
| `build/patch_v4n_mutator.py` | Pattern for the new `build/patch_supernext.py`. |
| `build/c3_gate.py`, `merge()` in C9, `p1_novel/*.parquet` | Reused by Wave C. |
| `EXPERIMENTS.md`, `specs/README.md`, new `specs/track4_priors_probes/{roadmap,gate_report}.md` | Track 4 documentation. |
| `updated_knowledge_base.txt` (repo root) | Knowledge base. |

## Verification

**Local, per variant:**
- Run `python .agents/skills/analyze-notebook/scripts/compare.py evenda-casmi-inferencesupernext.ipynb <variant>.ipynb`. The diff must show only the intended cell change.
- Add `build/test_patch_supernext.py`: check that each flag's code is present, the fallback is intact, and `IS_RERUN` is honoured. Run it via `sh build/run_all_tests.sh`.

**Kaggle commit run (12 molecules):** checks plumbing and cost only.
- The new counters print.
- The `pop stats` scale looks sane.
- Per-molecule time stays within the 4.5 h slack.

**Rerun / LB:**
- `submission.csv` is (400, 2), has no NaNs, and has ≤ 25 unique IK14 values per row.
- `top-1 changed in N` is greater than 0.
- Log every result with `/update-experiments`.

## Status (2026-10-02)

Step 0 is done:
- `updated_knowledge_base.txt` written.
- EXPERIMENTS row 24 (0.405) logged.
- specs README updated.
- `gate_report.md` pre-registered.

The Wave A generator is built: `build/patch_supernext.py` (65 checks in `build/test_patch_supernext.py`, wired into `run_all_tests.sh`).

**Wave A submission queue.** Upload these in order, each with the same datasets attached as supernext.

| order | notebook | probe |
|---|---|---|
| 1 | `casmi26-t4-a1precut.ipynb` | A1 prior before the TOPN cut |
| 2 | `casmi26-t4-a2popfused0.15.ipynb` | A2 prior on the fused list |
| 3 | `casmi26-t4-a4c1freeze.ipynb` | A4 class-1 top-1 freeze |
| 4 | `casmi26-t4-mu0.3.ipynb` / `casmi26-t4-mu0.08.ipynb` | A3 μ sweep |
| — | `casmi26-t4-a5genonly.ipynb` | A5, only to fill a spare slot |

**Stacking.** Winners combine through the flags, for example `python build/patch_supernext.py --pop-precut --c1-freeze`. A stacked notebook is itself a probe and must beat the best single change.

**Wave B blocker.** To map PubChem-tier entries to CIDs, we need the layout of the attached PubChem tier dataset (`pc_smiles` / `pc_off` — does it carry CIDs?). Inspect it on Kaggle before writing `build/prior_v2_colab.ipynb`.
