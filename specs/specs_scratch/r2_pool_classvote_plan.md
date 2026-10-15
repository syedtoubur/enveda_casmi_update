# specs_scratch R2 — class vote in tier-2 ranking (grilling outcome + build log, 2026-10-07)

Status: **R2.0 FAIL → fixed; R2.0b PASS (parity 50/50, 49/50 vs table, 14/14 known); R2.1 PASS (778,095 labelled, 0.970 table agreement); R2.2 PASS (purity 0.730 vs 0.139, coverage 0.948); R2.3 PASS (truth above median 0.802, up 122 / down 91, MRR@25 0.269 → 0.299 at sv = 0)**; **R2-GL.LB PASS 0.311 vs control 0.303 (+0.008), kept; R2-ICE pending** (sections 12, 14).
Gates are pre-registered in `gate_report.md` (R2 section). Next: the LB day (R2-ICE, R2-GL + same-day controls r1b, r1b_glacier),
after pool v2 is on Kaggle (see *Run order*).

## 1. Context — where R1 left us

| Run | LB | Note |
|---|---|---|
| Track 3 base (`casmi_simple_sim_submission.ipynb`, MAX_PEAKS=100) | 0.296 | reference |
| `casmi_scratch_r0` (own package port) | 0.290 | R0.4 FAIL (−0.006, bar 0.292) |
| `casmi_scratch_r1b` (R1b pool 597k, ICEBERG) | 0.301 | R1.LB PASS |
| `casmi_scratch_r1a_glacier` | 0.301 | |
| `casmi_scratch_r1b_glacier` (R1b, GLACIER) | 0.303 | best own scratch run; vs r1b +0.002 = noise |

The R1 class-vote CV diagnostic (slice I, 245 enveda-np-examples molecules, own-ik14 library spectra banned) passed:
the delta variant (direct ±10 ppm + NP biosynthetic delta list, shift-aware entropy, top 10 hits with sim ≥ 0.5,
similarity-weighted) voted the truth's superclass with **purity 0.878 vs majority baseline 0.168 at coverage 0.929**
(pathway purity 0.905). The open ±200 Da variant also passed (0.776 / 0.910) but costs ~41 min CPU vs ~30 s, so by the
pre-registered rule R2 uses the delta variant.

Caveats R2 has to survive (recorded in `kb_index.md`):
1. **Purity ≠ discrimination.** Isomers in one ±8.5 ppm window often share a superclass, so a correct vote may not
   separate the truth from its neighbours.
2. **Label coverage.** In the v1 pool ~66% of structures had no table/kNN label, so a label-matching bonus would have
   been silent on most candidates.
3. **Label leakage.** kNN labels could be inherited from the truth's own structure.
4. **Slice density.** Slice I is 230/250 in MassSpecGym, so its analog families are dense in the library — the CV is
   optimistic.
5. Prior art: Track 4b "class consistency" (PubChem channel only) was neutral → R2 gets a kill gate before any LB build.

## 2. Process

Requirement from the user: "use the class flags and class voting while ranking the candidate pool". The plan was
aligned with `/grill-me` in five rounds of clickable `AskUserQuestion` forms (preference saved to memory
`grilling-via-clickable-questions`). Free-text answers redirected the tree twice:
- Round 2: instead of deciding how to treat unlabelled candidates, the user proposed **labelling every compound**
  (Approach 1: NPClassifier NN; Approach 2: in-house LR/LightGBM on LOTUS's 152k labels). That opened Round 3.
- Round 4: "now keep the np score" was ambiguous; a clarifying round confirmed it meant **not** keep.

The approved plan file: `C:\Users\h\.claude\plans\here-inside-the-specs-scratch-jiggly-knuth.md`.

## 3. Decisions (settled)

| # | Topic | Decision | Alternatives considered |
|---|---|---|---|
| 1 | R0.4 debt | Diff R0 vs the Track 3 base **in parallel**, not blocking R2 | block R2 until the port is fixed |
| 2 | Base | `scratch_pkg/notebooks/casmi_scratch_r1b.ipynb` (R1b pool). **Two probes**: R2-ICE (`sim_model="iceberg"`) and R2-GL (`sim_model="glacier"`). Both simulators stay alive; decide later | only one simulator |
| 3 | Kill gate | Yes — a pre-registered discrimination check, **reach-based rule** (R2.3) | go straight to LB |
| 4 | Signals | Class vote + NPClassifier superclass labels only. **No NP_Score**, no provenance prior (deferred) | vote + NP_Score as a second notebook (dropped in Round 5) |
| 5 | Mechanism | **Additive bonus on sv** in the tier-2 sort key | multiplicative; filter; separate tier |
| 6 | Where | **Tier 2 only.** Tier 1 (lv ≥ τ = 0.75) untouched; stage-C mass-nearest top-up untouched | everywhere |
| 7 | Label level | **Superclass.** Class-level (hierarchical) matching is a later rung — class-level purity was never gated | pathway; class |
| 8 | Labeller | **NPClassifier** (Kim et al., J. Nat. Prod. 2021; mwang87/NP-Classifier, MIT; weights Zenodo 5068687) on Colab, after a spike. **Fallback** = logistic regression on the NP-table superclass labels | in-house LR/LightGBM first |
| 9 | Coverage | All **778,095 R1a structures** (superset of R1b and of every library structure) | R1b only; window candidates only |
| 10 | Precedence | **Model labels everywhere** — the same labeller on the vote side (library analogs) and the candidate side. Table/kNN labels kept only as audit columns | table first, model as fill-in |
| 11 | Match | **Soft**: `P_cand(voted)` from the candidate's top-3 NPClassifier superclass probabilities (0 if absent) | hard top-1 match |
| 12 | β | **0.10 fixed**, pre-registered, never tuned | sweep |
| 13 | sv = 0 rows | **Bonus applies anyway** (GLACIER's non-[M+H]+/[M+Na]+ spectra, unsimulated candidates) — the vote then leads fr; counted by `n_vote_first` | skip |
| 14 | CV re-run | Same rule as R1 (purity ≥ baseline + 0.10 and coverage ≥ 50%) | |
| 15 | LB day | R2-ICE, R2-GL + same-day paired controls r1b and r1b_glacier | GL control + 2 GL + 2 ICE (superseded once NP_Score was dropped) |

## 4. Mechanism (exact)

Per molecule (pass 1, `pipeline.plan_molecule` → `molecule_vote`):
1. `analog.analog_hits(lib, specs, target, mode="delta", polarity=same, exclude=own-ik14 ban, ppm=cfg.vote_ppm=10)`:
   library spectra whose precursor sits at target ± a delta from the NP list (CH₂, O, H₂O, CH₂O, C₂H₂O, CO₂, C₅H₈,
   C₃H₂O₃, C₅H₈O₄, C₉H₆O₂, C₆H₁₀O₄, C₆H₁₀O₅, C₆H₈O₆, plus 0), scored by shift-aware entropy.
2. `analog.class_vote(hits, VOTE_LABEL.get, k=10, min_sim=0.5)`: the top 10 hits with sim ≥ 0.5 vote with their
   similarity as weight for their structure's **top-1** NPClassifier superclass.
   `VOTE_LABEL = pool.npc_top1(POOL_DIR, set(LIB.ik))` — read from the whole pool file, so library structures dropped
   from R1b keep their labels.
3. `conf = analog.vote_conf(w, win)` = winner weight / total voting weight.

Per window (pass 2, `rank_molecule`, stages A and B):

```
bonus_i = β · conf · P_i(voted)          # analog.vote_bonus, P from sc1..3/sp1..3, 0 if voted ∉ top 3
tier 1  : lv ≥ τ   → lexsort by lv, sv, fr, |ppm|            (unchanged)
tier 2  : lv < τ   → lexsort by (sv + bonus), fr, |ppm|      (rank.rank_two_tier(..., bonus))
```

`bonus=None` reproduces the R1 order exactly (tested). With β = 0.10 and sv on an entropy scale of 0–1, the bonus is at
most 0.10: it reorders near-ties in sv and dominates only where sv = 0.

Counters in the run report (`pipeline.vote_counters`): `n_vote_mol` (molecules with a vote), `n_bonus_reorders` (top 25
differs from the bonus-free order), `n_vote_first` (sv max = 0 but a positive bonus, i.e. the vote leads). `run()`
raises if `class_vote=True` but no labels were passed.

## 5. Pre-registered gates (copied from `gate_report.md`, written before any measurement)

| Gate | Threshold | On fail |
|---|---|---|
| **R2.0 spike** | NPClassifier weights load on Colab; fingerprint parity with the repo's `calculate_fingerprint` on 50 random table-labelled rows (50/50); model top-1 superclass ∈ table label for ≥ 45/50; 10k SMILES timed | **FAILED 1/50** (RDKit bit drift, section 12) |
| **R2.0b re-run** | Same thresholds; fingerprints from RDKit 2021.03.5 (side env); parity checked in that env; 14 known structures reported (not gated) | Switch to `LABELLER = "lr"` (report holdout accuracy vs table) |
| **R2.1 labels** | Every R1a row RDKit parses gets top-3 superclasses; unparsable count logged; top-1 agreement with table reported | — |
| **R2.2 CV re-run** | Slice I, own-ik14 ban, delta variant, NPClassifier top-1 labels on both sides: purity@superclass ≥ majority baseline + 0.10 **and** coverage ≥ 50%; table-only purity reported | Idea weakened; discuss before LB |
| **R2.3 discrimination (KILL)** | Slice I, own-ik14 ban, scored range, **sv = 0** (no simulator on Colab; fr is the tier-2 base — the worst case, as on GLACIER's no-sim spectra), β = 0.10. PASS iff (1) truth's P(voted) > window median P(voted) in ≥ 60% of voted molecules **and** (2) the bonus moves the truth up in more molecules than down. Reported: reach (% top 25 changed), MRR@25 before/after, median top-1/top-2 fr gap | **No LB build**; recorded as a dead end |
| **R2.run** | Fast commit: `n_vote_mol` > 0, `n_bonus_reorders` > 0, `sim_ok_share` ≥ 0.99, 0 capped/failed rows; full run ≤ 7.5 h | Fix before LB |
| **R2.LB** | Same day: R2-ICE vs r1b, R2-GL vs r1b_glacier. Keep each iff ≥ its control + 0.004 and counters > 0 | Not kept |

Why sv = 0 in R2.3: R1.2 was uninformative and the audit has no simulator. Using fr only as the tier-2 base is the
harshest setting (the bonus competes with fr alone and moves the most rows), so a pass there is meaningful.

## 6. What was built

### Package (`scratch_pkg/casmi_scratch/`)
| File | Change |
|---|---|
| `npc.py` (new) | `fingerprint` (radius-0 counted Morgan → `input_2048`, radius-1/2 → `input_4096`, explicit H, exact-radius environments only; **verified identical** to the repo's `calculate_fingerprint` on 4 molecules), `fingerprints`, `label_names(index_v1.json)`, `load_models` (tf_keras first — the weights are Keras/TF 2.3 hdf5), `model_inputs` (by input width), `topk`, `decode` (sc1..3, sp1..3, pathway/pathway_p, class/class_p), `predict(..., fp_fn)` (refuses a post-2021.03 RDKit without an explicit `fp_fn`), `rdkit_bits_ok`, `fp_env_cmd` / `fp_subprocess` / `fp_fn_subprocess` (fingerprints from the old-RDKit worker, section 12), `label_all(..., fp_fn)` (chunked, resumable via `pool_build.run_chunked`, 20k/chunk), `write_npc` / `load_npc` (key-aligned `pool_npc.parquet`; codes int16 with −1 = none), fallback `lr_fit` / `lr_predict` (SGD log-loss on log1p counts) |
| `resources/npc_fp_worker.py` (new) | Standalone fingerprint worker (numpy + RDKit only) run under Python 3.9 + `rdkit-pypi==2021.3.5`; multiprocess, keeps row order, optional parity vs the repo's `calculate_fingerprint` (`--ref`), refuses RDKit > 2021.03 (`--any-rdkit` for tests) |
| `pool.py` | `CandidatePool(..., npc=None)` carries labels through the mass sort and `without()`; `load_pool(..., with_npc=False)`; `npc_top1(pool_dir, keys)` |
| `analog.py` | `vote_conf`, `vote_bonus` |
| `rank.py` | `rank_two_tier(..., bonus=None)` — tier-2 key `sv + bonus` |
| `config.py` | `class_vote=False`, `vote_beta=0.10`, `vote_k=10`, `vote_min_sim=0.5`, `vote_ppm=10.0` |
| `pipeline.py` | `molecule_vote`, `plan_molecule(vote_label=)` stores `vote=(label, conf)`, `class_bonus`, bonus in stage A and B, diag fields `voted / bonus_max / reordered / vote_first`, `run(vote_label=)`, `vote_counters` |
| `audit.py` | `discrimination(...)` (per-molecule rows: voted, truth P vs window median, truth rank before/after, reach, fr gap) and `summarize_discrimination(rows, topn)` → R2.3 verdict |

### Drivers and notebooks (generated by `scratch_pkg/tools/mk_notebook.py`, never hand-edited)
| Notebook | Driver | Runtime | Does |
|---|---|---|---|
| `casmi_scratch_r2_npc_colab.ipynb` | `drivers/r2_npc_colab.py` | Colab T4, internet ON | Downloads weights/index/`fingerprint_handler.py` to Drive `_npc/`; installs `uv`; R2.0b spike with old-RDKit fingerprints (cached in `spike_npc_r20b.json`; the failed R2.0 is in `spike_npc.json`; stops with `SystemExit` on fail); R2.1 labels all rows (checkpointed in `_npc/ckpt_fp2021/`, resumes); writes `pool_npc.parquet`, MANIFEST → `v2` with an `npc` block |
| `casmi_scratch_r2_audit_colab.ipynb` | `drivers/r2_audit_colab.py` | Colab CPU High-RAM | Asserts pool v2; R2.3 discrimination on the R1b pool; R2.2 `class_vote_diag(modes=("delta",))` with NPClassifier labels; table-only purity; verdicts → `_r2_audit/verdicts.json` |
| `casmi_scratch_r2_ice.ipynb` | `drivers/r0_kaggle.py` | Kaggle T4, offline | = r1b + `class_vote=True` (diff verified: only that switch and RUN_TAG) |
| `casmi_scratch_r2_gl.ipynb` | `drivers/r0_kaggle.py` | Kaggle T4, offline | = r1b_glacier + `class_vote=True` (same diff check) |

`r0_kaggle.py` now loads the pool with `with_npc=CFG.class_vote`, builds `VOTE_LABEL`, passes it to `pipeline.run`,
prints the vote counters and checks the vote fires. No TensorFlow on Kaggle — labels are read from the dataset.
All other notebooks were regenerated (the package is embedded); with `class_vote=False` they behave as in R1.

### Tests
`scratch_pkg/tests/test_r2.py` (12): bonus=None = R1 order; bonus reorders tier 2 only; `vote_conf` + soft bonus;
fingerprint; names/decode (unparsable rows); `pool_npc` I/O alignment; npc survives the mass sort; `class_bonus` +
counters; `run` fails without labels; `label_all` resumes from checkpoints (fake model); discrimination verdict;
end-to-end discrimination on `toy_analog_lib`. Plus `test_dryrun_r2`. After the RDKit fix, +2:
`test_npc_rdkit_bits_guard`, `test_npc_fp_worker_matches_in_process`. Full suite green: test_core 17, test_dryrun 3,
test_glacier_worker 2, test_notebook 4, test_r1 15, test_r2 14.

### Docs
`gate_report.md` (R2 section + R0.4 diff verdict), `roadmap.md` (S3 = R2, S4 = deferred), `data_contract.md`
(`pool_npc.parquet`, pool v2), `kb_index.md` (R2 note).

## 7. Data contract addition — `casmi26-scratch-pool` v2

`pool_npc.parquet`, row-aligned with `pool_key.npy` (mass order), with a `key` column checked on load:
`key, sc1, sc2, sc3` (superclass names, None if unparsable), `sp1, sp2, sp3` (f32 probabilities, sigmoid outputs of
the 76-way superclass head), `pathway, pathway_p, class, class_p` (top-1, kept for later rungs). `MANIFEST.json`
`version = "v2"` plus an `npc` block (labeller, counts, spike result, source, build time, RDKit version, sha1).

## 8. R0.4 diff (done in parallel, as agreed)

`build/simple_sim_submit.py` vs `casmi_scratch`, by hand (`compare.py` is useless here: R0 embeds the package as one
string). Target centring, pass 1 (lv, fr, covered spectra, sim set, priority), the watchdog, pass 2 stages A/B/C and
slot dedup are the same. The one material difference is planned: R0 cleans query spectra for `sv` at `max_peaks` 100,
the base used `simsim_common.TOPK` = 64. **No port bug.** The −0.006 is that change or noise (1.5× the floor). An
`sv_max_peaks=64` probe would separate the two — not built.

## 9. Run order (user-side)

1. **Colab, T4, internet ON:** the **regenerated** `casmi_scratch_r2_npc_colab.ipynb` (R2.0b). Paste the printout. If
   R2.0b fails, set `LABELLER = "lr"` in the first cell and re-run. After a disconnect, re-run all cells (it resumes).
2. **Kaggle:** upload `casmi_2026/casmi26-scratch-pool/` as **version 2** of `casmi26-scratch-pool`. The R2 notebooks
   fail without `pool_npc.parquet`.
3. **Colab, CPU High-RAM:** `casmi_scratch_r2_audit_colab.ipynb`. Paste the R2.2 / R2.3 verdicts → recorded in
   `gate_report.md`.
4. **Only if R2.3 passes:** same day, submit `casmi_scratch_r2_ice`, `casmi_scratch_r2_gl`, `casmi_scratch_r1b`,
   `casmi_scratch_r1b_glacier`. Check R2.run counters on the fast commit. Log via `/update-experiments`.

## 10. Risks

- **Keras version.** Resolved: `tf_keras` loads the TF 2.3/2.4 hdf5 files on Colab (R2.0 printout).
- **Side env on Colab.** `uv` must fetch CPython 3.9 and the `rdkit-pypi` 2021.3.5 manylinux wheel (resolution checked
  locally for x86_64-manylinux_2_17; not yet run on Colab). The first fingerprint call builds the env.
- **NPClassifier vs table disagreement.** Table labels come partly from NPClassifier itself, so R2.0 agreement is
  expected to be high; a low score means a fingerprint or name-index bug, not a weak model.
- **sv scale.** β = 0.10 is meaningful only relative to the spread of sv near the tier-2 top; it is fixed by
  pre-registration and not tuned on the LB.
- **Optimistic slice.** Slice I analog families are dense in the library; the LB is the only real judge.

## 11. Deferred (roadmap S4)

NP_Score prior; provenance prior; class-level (hierarchical) matching; pathway-level vote; β sweep (only with a new
pre-registration); GLACIER/ICEBERG hybrid; `sv_max_peaks=64` probe for R0.4; r1a (ICEBERG) LB pair still pending.

## 12. R2.0 FAIL and the fix (2026-10-07)

**Symptom (Colab):** weights loaded (heads 7 / 77 / 687), fingerprint parity 50/50, but top-1 ∈ table label only
**1/50**, with nonsense misses (a steroid labelled "Nicotinic acid alkaloids"). That is far below the ~17% from always
guessing the most common class, so this was a bug, not a weak model.

**Diagnosis (local, no TensorFlow):** downloaded the Zenodo weights and ran a numpy forward pass of the superclass
net on 14 well-known natural products.
- The label index was ruled out: `index_v1.json` is clean (names 0..76 in order, matches the 77 outputs).
- TF loading was ruled out: numpy gives the same noise.
- Input variants (no explicit H, binarised counts, swapped blocks) were all worse. The unchanged encoding matched the
  BatchNorm statistics best (corr 0.60), but not well.
- **Root cause:** RDKit changed the Morgan bit assignment between **2021.03.5** and **2021.09.5**. Fingerprints are
  identical for 2020.09.5 and 2021.03.5, and identical for 2021.09.5, 2022.03.5 and 2026.03.3; the two groups share
  few bits. NP-Classifier V1 was trained on the old bits.
- With RDKit 2021.03.5 fingerprints the net labels all 14 correctly, mostly at p ≈ 1.0, and first-layer activations
  correlate 0.945 with the BatchNorm means.
- The R2.0 parity check could not catch this: it compared our code against the repo's code on the same new RDKit.

**Fix:**
- `resources/npc_fp_worker.py` (new): a standalone worker that computes the fingerprints in a side env (Python 3.9 +
  `rdkit-pypi==2021.3.5` via `uv run`). It refuses any RDKit after 2021.03 and can check parity against the repo's
  `calculate_fingerprint` in that same env.
- `npc.py`: `fp_subprocess`, `fp_fn_subprocess`, `rdkit_bits_ok`. `predict` and `label_all` take `fp_fn`; without one,
  `predict` refuses to run on a new RDKit. Empty SMILES are now unparsable in both paths.
- Colab driver: the R2.0b spike reports parity from the old env, 50-row table agreement and the 14 known structures.
  It uses new cache names (`spike_npc_r20b.json`, `ckpt_fp2021/`), so the failed spike is not reused. The MANIFEST
  `npc` block records `fp_rdkit`.
- Tests: `test_npc_rdkit_bits_guard`, `test_npc_fp_worker_matches_in_process` (multiprocess row order, unparsable rows,
  version refusal). Suite green: test_r2 14, all others unchanged.
- Verified locally: the worker under RDKit 2021.03.5 gives parity 14/14 with the repo code and the same 14 correct
  labels; `uv` resolves the env for Linux / Python 3.9.

**Gate bookkeeping:** the R2.0 FAIL stays recorded. R2.0b (same thresholds) is pre-registered in `gate_report.md`
before the re-run. The LR fallback applies only if R2.0b fails.

## 13. Local test assets (kept on purpose, 2026-10-07)

Downloaded weights, the NP-Classifier repo files and three old-RDKit conda envs. They are kept on the user's machine for
later tests of the labeller. None of it is in Git; everything is outside the repo.

> **Volatile locations.** Everything below is under `%TEMP%` (`C:\Users\h\AppData\Local\Temp`), which Windows Storage
> Sense or a disk clean-up can empty. The scripts in the session scratchpad (13.4) are session-scoped and are the most
> likely to vanish. If something is missing, rebuild it with the commands given.

### 13.1 NPClassifier weights — `C:\Users\h\AppData\Local\Temp\npc_dl\` (≈ 953 MB)
| Path | What |
|---|---|
| `npc_dl\model.zip` | Zenodo 5068687 `model.zip`, complete (732,938,302 bytes): `NP_classifier_{class,pathway,superclass}_V1.hdf5` (Keras 2.4.0 / TF backend) |
| `npc_dl\models\NP_classifier_superclass_V1.hdf5` | Extracted superclass head only (77 sigmoid outputs). Extract the others with `python -I -c "import zipfile; zipfile.ZipFile('model.zip').extractall('models')"` |

Re-download: `curl -L -C - -o model.zip "https://zenodo.org/record/5068687/files/model.zip?download=1"`. The first
download stopped silently at 606 MB; `-C -` resumes. Check the size before unzipping.

### 13.2 NP-Classifier repo files — `C:\Users\h\AppData\Local\Temp\npc_idx\`
From `github.com/mwang87/NP-Classifier` (master, MIT):
| File | Repo path | Use |
|---|---|---|
| `index_v1.json` | `Classifier/dict/index_v1.json` | label names: Pathway 7, Superclass 77, Class 687 (name → output index, alphabetical 0..n−1), plus `Class_hierarchy` / `Super_hierarchy` |
| `fingerprint_handler.py` | `Classifier/fingerprint_handler.py` | reference `calculate_fingerprint` (parity checks; needs pandas + tqdm to import) |
| `train_fp.py` | `training/fingerprint_handler.py` | training-side copy; same fingerprint code (only `_isglycoside` differs) |
| `app.py`, `prediction_voting.py` | `app.py`, `Classifier/prediction_voting.py` | how the web app thresholds (superclass ≥ 0.3, class ≥ 0.1, pathway ≥ 0.5) and votes |
| `Model.py`, `conv.py` | `training/Model.py`, `Classifier/models_folder/models/convert_keras_to_tf.py` | architecture (Concat 6144 → Dense/BN … → sigmoid) and the hdf5 → SavedModel export |

### 13.3 Old-RDKit conda envs — `C:\Users\h\AppData\Local\Temp\rdk*\`
All Python 3.8.20 (conda-forge) with numpy, pandas, tqdm. Run with `<env>\python.exe -I <script>`.
| Env | RDKit | Morgan bits |
|---|---|---|
| `C:\Users\h\AppData\Local\Temp\rdk2020\` | 2020.09.5 | **old** (what NP-Classifier V1 was trained on) |
| `C:\Users\h\AppData\Local\Temp\rdk2021.03\` | 2021.03.5 | **old**, identical to 2020.09.5; same version as the Colab side env |
| `C:\Users\h\AppData\Local\Temp\rdk2021.09\` | 2021.09.5 | new, identical to 2022.03.5 and 2026.03.3 |

Rebuild: `conda create -y -p %TEMP%\rdk2021.03 -c conda-forge python=3.8 rdkit=2021.03 numpy pandas tqdm`.
`rdkit-pypi==2021.3.5` has no Windows wheel (Linux/macOS only), so locally use conda. RDKit 2022.03.5 was tested through
`uv run --python 3.9 --with rdkit-pypi==2022.3.5` (no env kept). The main interpreter (miniconda, RDKit 2026.03.3) also
has `h5py` 3.16 installed for the numpy forward pass.

Worker under the old RDKit, e.g. from `scratch_pkg/`:
```python
from casmi_scratch import npc
F, B, ok, meta = npc.fp_subprocess(smiles, cmd=[r"C:\Users\h\AppData\Local\Temp\rdk2021.03\python.exe", "-I",
                                   str(npc.FP_WORKER)], workers=2, ref_dir=r"C:\Users\h\AppData\Local\Temp\npc_idx")
# meta = {'rdkit': '2021.03.5', 'parity': <rows equal to the repo's calculate_fingerprint>}
```

### 13.4 Diagnostic scripts and fingerprint dumps — session scratchpad
`C:\Users\h\AppData\Local\Temp\claude\E--ENVEDA-casmi\59f19e3a-c614-491d-9e4c-e93a8b8292f6\scratchpad\`
| File | What | Run |
|---|---|---|
| `fp_dump.py` | fingerprints of the 14 known structures with the repo's `calculate_fingerprint` → npz (`X` (14, 6144), `names`, `ver`) | `<env python> -I fp_dump.py OUT.npz %TEMP%\npc_idx` |
| `fwd_npz.py` | numpy forward pass of a superclass hdf5 on npz dumps; prints layer-1 activation vs BatchNorm means and the top 2 per structure | `python -I fwd_npz.py <superclass.hdf5> <index_v1.json> A.npz [B.npz …]` |
| `npc_variants.py` | the same forward pass over input variants (as is / no H / binary / swapped blocks) on the current RDKit | `python -I npc_variants.py <superclass.hdf5> <index_v1.json>` |
| `npc_fwd.py` | first forward-pass probe (5 structures, uses `casmi_scratch.npc.fingerprint`) | `python -I npc_fwd.py <hdf5> <index_v1.json> <scratch_pkg dir>` |
| `fp_2020.npz`, `fp_2021.03.npz`, `fp_2021.09.npz`, `fp_2022.3.5.npz`, `fp_new.npz` (2026.03.3), `fp_worker.npz` (worker under 2021.03.5) | the 14-structure dumps per RDKit version | inputs to `fwd_npz.py` |

The 14 structures and their expected superclasses are listed as `SANITY` in `scratch_pkg/drivers/r2_npc_colab.py`.

### 13.5 Quick re-checks
- **Is a fingerprint path right?** Dump the 14 structures with it and run `fwd_npz.py`. All 14 should be correct, most
  at p ≈ 1.0, with layer-1 correlation ≈ 0.95. New-bits fingerprints give noise and a correlation ≈ 0.60.
- **Does an RDKit version give old or new bits?** Compare its `fp_dump.py` output with `fp_2021.03.npz`
  (`np.array_equal`).
- **Package tests:** `python scratch_pkg/tests/run.py` (none of them needs these assets).

**Result on Colab (2026-10-07):** R2.0b PASS: fingerprint env RDKit 2021.03.5, parity 50/50, top-1 ∈ table 49/50,
14/14 known structures at p ≥ 0.988, 4.2 s per 10k. R2.1 PASS: 778,095 / 778,095 labelled, 0 unparsable, top-1
agreement with the 131,644 table rows 0.970, sp1 median 0.710, 5 min. Open observation: alkaloid superclasses lead the
top-1 counts (Lysine 68k, Tryptophan 66k, Nicotinic acid 56k), probably out-of-domain argmax on non-NP N-heterocycles.
The P-weighted bonus damps it on the candidate side; the vote side is tested by R2.2 / R2.3.

## 14. R2.2 / R2.3 audit results (2026-10-07)

| Gate | Result | Bar |
|---|---|---|
| R2.2 CV | coverage 0.948 (237/250), purity 0.730 vs majority baseline 0.139 (Flavonoids); pathway 0.848 vs 0.316; table-only purity 0.816 (n 190) | purity ≥ baseline + 0.10, coverage ≥ 0.50 → **PASS** |
| R2.3 kill | truth P(voted) > window median in 0.802 of voted; up 122 / down 91; reach 0.928; MRR@25 0.2691 → 0.2987; median fr gap 0.017 | ≥ 0.60 and up > down → **PASS** |

How to read it:
- **Purity fell from 0.878 (R1) to 0.730.** R1 matched against table/kNN truth labels; a multi-valued label (`A|B`)
  counted a hit on either value, and kNN labels could be inherited from the truth itself. NPClassifier top-1 on both
  sides has neither advantage, so 0.730 is the more honest figure. It is still about 5× the baseline.
- **R2.3 is a worst case for size, not a forecast.** With sv = 0 the bonus (≤ 0.10) is about 6× the median fr gap
  (0.017), so it effectively becomes the tier-2 sort key: hence reach 0.93. On Kaggle, sv separates candidates first and
  the bonus mostly breaks near-ties, so expect smaller reach and a smaller gain. GLACIER's no-sim spectra (other
  adducts) are the exception: they rank like this audit.
- **The bonus cuts both ways.** 91 of 213 moved truths went down. The +0.030 MRR is a net of large moves in both
  directions, so LB noise matters and the paired control is essential.
- **Slice I is optimistic.** It is 230/250 in MassSpecGym, so analog families are dense in the library. On the test
  set, vote coverage (0.948 here) and purity will be lower; `n_vote_mol` in the Kaggle run reads the real coverage.
- The trailing `593` in the printout is the number of characters `write_text` wrote for `verdicts.json`, not a metric.
