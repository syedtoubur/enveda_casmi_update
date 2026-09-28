# specs_novel — gate report

*The thresholds for each gate are written here and committed **before** the gate's measurement runs
(roadmap protocol 1). Results are filled in afterwards, and each result cites the notebook and stage
that produced it. A failed gate stops the project (protocol 2).*

---

## G0 — Feasibility (P0)

*Pre-registered 2026-09-28, spec v3.1. Not run yet.*

Sources, once run:

- `casmi_novel_p0_colab.ipynb` writes `_novel/p0/phase0_colab_gate.json` (P0.1a, P0.2–P0.5).
- `casmi_novel_p0_kaggle.ipynb` writes `phase0_kaggle.json` (P0.1b; T4, internet off).

### Verdict: **PENDING**

| gate | pass condition (pre-registered) | result | status |
|---|---|---|---|
| G0.1 | **Kaggle, offline:** `dreams_embed.pt` loads with its SHA1 asserted, and cosine ≥ **0.999** on **all 200** parity rows, Kaggle (b) vs Colab (b). **Colab:** official (a) vs vendored (b) ≥ **0.9999** on all 200 rows, and vendored (b) vs cache (c) ≥ **0.999** on all 200 rows. | — | PENDING |
| G0.2 | Replaying `casmi_dreams_colab.ipynb` cell 7 reproduces `corpus_meta.parquet` **row by row**: `ingest_lib`, `inchikey14` and `adduct` equal, `precursor_mz` equal to float64. `corpus_ce.parquet` is written. | — | PENDING |
| G0.3 | The SELFIES round-trip (stereo and isotopes stripped → SELFIES → SMILES → ik14, RDKit 2026.03.3) is ≥ **0.99** of the unique corpus structures that are single-component and contain only C, H, N, O, P, S, F, Cl, Br and I. The excluded share is reported, and `MAX_LEN` is set at p99.9 of the token lengths. | — | PENDING |
| G0.4 | Fast-tracker vs reference-tracker parity is **100%** on every prefix of 10k random structures (remaining counts, unsaturation, atom placed). Mask soundness over **all** round-trip-OK structures shows **0 violations** of the count, `</s>` and length rules. | — | PENDING |
| G0.5 | `SelfiesTokenizer`, `MolToFormulaVector`, `DeNovoMassSpecGymModel` and `RandomDeNovo` import on Colab Python 3.12, and `msg_ik14.npy` is built. | — | PENDING |

### Pre-registered decision rules

1. **G0.1, cache leg.** If (b) vs (c) fails while (a) vs (b) passes, the cache was built by a
   different DreaMS commit (A6). The gate is then **HOLD, not FAIL**. The drift is diagnosed and a
   re-embed of the corpus with the pinned commit is costed and scheduled. G0.1 passes once the
   re-embed exists and both other legs pass.
2. **G0.2.** A row mismatch is HOLD in the same way: a re-embed with CE stored is costed and
   scheduled (roadmap P0.2).
3. **G0.4, DBE prune (A8).** Any DBE-prune violation sets `DBE_PRUNE = False` and is recorded with
   examples. It **does not** fail G0.4. Only count, `</s>` or length violations fail it.
4. **G0.4, tracker.** A parity failure that cannot be fixed falls back to the reference tracker
   (plan §6). G0.4 then passes only if the reference tracker's mask walk shows 0 violations, and the
   slowdown is recorded for P4.
5. **Anything else that fails** stops the project at P0, as roadmap protocol 2 requires.

### Recorded, not gating

| item | value | source |
|---|---|---|
| SHA1 `embedding_model.ckpt` / `ssl_model.ckpt` / `dreams_embed.pt` | — | P0.1a |
| DreaMS head fine-tuning data (leakage note; provisionally MoNA `[M+H]+`, `fine_tune.sh:45-71`) | — | P0.1a |
| `T_novel_install` (Kaggle) | — | P0.1b |
| `T_embed` per spectrum (T4, `cuda:1`) | — | P0.1b |
| corpus rows / unique ik14 / per-adduct / mode split / share m/z > 1000 | — | P0.2 |
| CE histogram, missing-CE share, `CE_BINS` | — | P0.2 |
| round-trip failure classes; excluded share (multi-component, disallowed elements) | — | P0.3 |
| `MAX_LEN`; share of structures above it | — | P0.3 |
| SHA1 `selfies_vocab.json`; vocabulary size | — | P0.3 |
| `selfies` pinned version and licence | — | P0.5 |
| DBE-prune violations (count, examples) and the `DBE_PRUNE` decision | — | P0.4 |
| P1 preview: `|corpus ik14 \ msg_ik14|`, overall / NP libraries / enveda-np-examples | — | P0.5 |

### Findings

*(filled after the run)*
