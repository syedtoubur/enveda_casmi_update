# Mission — Simple MS Simulator Baseline

*Revision 1 — 2026-09-26. Companion files: `roadmap.md` (phases and checks), `tech_stack.md`
(technical details). The broader GBDT-channel plan lives in `specs_ms_simulator/` and is not
affected by this spec.*

## 1. Goal

Build a **lean, standalone, offline Kaggle submission** for Enveda CASMI 2026 (top-25 SMILES per
molecule, scored MRR@25 on InChIKey14). It turns **Class 2 into Class 1 through a virtual
library**: every candidate structure that lacks a real reference spectrum gets a *simulated* one,
and the query is matched against it as if it were a library spectrum.

This is the **reference baseline** that later, more complex approaches build on. It does **not**
have to beat the 0.337 incumbent. It has to be correct, run offline in ≤ 9 h, and produce a
leaderboard number that is easy to interpret.

## 2. Why a separate baseline

The 0.337 pipeline (library + analog + MetFrag-lite + FPNet → 31-column GBDT) is tightly coupled.
12 of its 31 ranker features are FPNet logits, so any change means a Stage C/D rebuild. It is
also hard to attribute gains to individual channels. This baseline has **no learned ranker** and
two scoring sources, so its score shows directly how much a forward simulator is worth.

## 3. Outlined approach

```
 query molecule (1-16 spectra, one molecule_id)
        |
        v
 neutral mass  = median over spectra of neutral_mass(precursor_mz, adduct)
                 / (1 + PPM_CENTER*1e-6),  PPM_CENTER = 1.45 (timsTOF offset)
        |
        v
 candidate window = pool.window(target, 8.5 ppm)   (~40-200 candidates)
                    + PPM_EXPAND tiers / backfill if < 25 filled   (phase1 logic, unchanged)
        |
        +--------------------------+-----------------------------+
        v                                                        v
 TIER 1 - real library                               TIER 2 - virtual library
 lv = max entropy sim(query, real spectra of        sv = max entropy sim(query spectrum,
      that ik14)   [phase1 lib_sim]                      simulated spectrum of candidate
 candidates with lv >= tau                               at the query's adduct + CE)
 ordered by lv desc                                 uncovered adduct/mode -> MetFrag-lite
                                                    explain score instead
        |                                                        |
        +--------------------------+-----------------------------+
                                   v
            ranked list = tier 1 (by lv) then tier 2 (by sv, tie-break MetFrag, ppm error)
                                   v
                      InChIKey14 dedup -> top-25 SMILES
```

- **Two-tier ordering.** A real-vs-real similarity and a simulated-vs-real similarity are on
  different scales, so they are never compared directly. A confident real match (`lv ≥ τ`) always
  comes first. Everything else, including candidates with a weak real match, is ranked by its
  simulated score.
- **Simulator.** Pretrained **ICEBERG** from `coleygroup/ms-pred`, using **MassSpecGym-trained
  checkpoints only** (MIT licence). NIST-trained weights are forbidden.
- **Where simulation happens.** Both places:
  1. **Precomputed.** A simulated library for the *entire* candidate pool is built on a small
     (adduct × collision-energy) grid in Colab and shipped as a Kaggle dataset. It covers
     structures with *and* without real spectra.
  2. **On the fly.** Kaggle simulates only when a query's condition falls outside the grid
     (e.g. CE 120 eV) or a candidate is missing from the precomputed library.
- **Multi-spectrum aggregation.** Take the max over the molecule's spectra, the same as phase1
  `lib_sim`.
- **Uncovered spectra.** For adducts or ion modes ICEBERG cannot simulate (expected: negative
  mode and several positive adducts), tier 2 falls back to the existing MetFrag-lite score.

## 4. Secondary goal — a simulator error atlas

Simulating structures that *do* have real spectra costs little extra, and it gives a paired
(simulated, real) dataset. That dataset shows where the simulator fails: by compound class,
mass, adduct, collision energy and source library. It is stored for later specs, and **nothing is
gated on it**.

## 5. Scope

| in scope | out of scope |
|---|---|
| Class 1 via real library search | Class 3 / de novo generation |
| Class 2 via simulated library over the existing pool | Pool expansion (LOTUS, NPAtlas, PubChem-NP) |
| MetFrag-lite fallback for uncovered spectra | FPNet, analog channel, GBDT ranker, `rank_train.npz` |
| One direct LB submission (plus cheap τ variants) | `val_scripts` / Stage-D validation |
| | Fallback simulators — if ICEBERG fails its gate, **stop and re-plan** |

## 6. Standing rules carried over

- Correctness is InChIKey14 after RDKit **2026.03.3** tautomer canonicalisation. Dedup on ik14
  before emitting.
- Only freely and publicly available checkpoints. Record the SHA1 and assert it at load.
- Notebooks are **generated** from `build/mk_*.py`, never hand-edited.
- Reused code is taken from `enveda-casmi-2026-phase1-enhanced-baseline.ipynb`.
- Every number in these specs is either **measured** (with its source) or marked **to measure**.
