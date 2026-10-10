# Knowledge-base index: lever → evidence

Every rung must cite a row here. Deltas are LB, from `EXPERIMENTS.md` unless noted.

## Levers that paid
| Lever | Evidence | Delta | Notes for our rebuild |
|---|---|---|---|
| ICEBERG forward sim | rows 01→02 | +0.035 | Our base already has it. |
| `MAX_PEAKS` 64→100 (lv/fr only) | own run, not in ledger | +0.007 | R0 extends it to `sv`. |
| BIO pool (ChEBI + LIPID MAPS) | Track 1 | +0.009 | R1. |
| Popularity in retrieval (apex v4b) | Track 4 | +0.010 | R2 candidate. Re-order a shortlist; choosing the shortlist by popularity is unproven (A1 neutral). |
| GLACIER as 3rd scorer, λ 0.5→1.0 | Track 1 | +0.007, +0.007 | R2 candidate. `[M+H]+` / `[M+Na]+` only. Being probed now as an ICEBERG *replacement* (R1-GL twins, `gate_report.md`). |
| Engine RRF (α 0.6) | Track 1 | +0.016 | R5 candidate: needs a 2nd orthogonal engine. |
| ICE_UNION | Track 1 | +0.004 | Simulators score both engines' pools. |

## Dead ends (do not repeat)
λ ≥ 2 (−0.034); tuning on local val (−0.015); FPNet trained on ranker queries (−0.051); TOPN 60→100 (0.000);
FPNet backbone swap (−0.002); absolute-threshold gates (−0.064); generated candidates in RRF (0.000/−0.002);
NP-likeness / class consistency (0.399 vs 0.400); every Class-3 lever (±0.002, ceiling ≈ +0.010).

## R2 candidates from the R1 grilling (10-05)
- **Class vote**: analog library hits (direct + NP delta list, or open ±200 Da; shift-aware entropy) vote an NP superclass;
  boost same-class pool candidates. Unlike Track 4b class consistency (PubChem channel only, oracle analogues), it acts on
  the main pool list, so reach is large. Gated by the CV diagnostic in `gate_report.md` before any build.
  **CV PASS 10-06**: delta-list superclass purity 0.878 vs baseline 0.168, coverage 0.93, so R2 uses the delta variant.
  Before building, measure **discrimination**: the share of the truth's scored-window candidates that carry the voted
  superclass, and how often the truth is labelled at all. If most window candidates share the class (isomers often
  do), the boost cannot move the truth. Also report purity with table-only labels, since a kNN label can come from
  the truth itself.
  **R2 built 10-07** (`gate_report.md` R2): labels are NPClassifier itself on all 778k structures (LOTUS's labels are
  NPClassifier output, so a table-trained classifier would only copy it); soft match on top-3 probabilities; β = 0.10.
  The discrimination check above is now the pre-registered kill gate R2.3.
- **Drop enveda-180-only structures** from the pool (they are synthetic decoys for an NP test, yet their own spectra give
  them high `lv`): probed as R1b.

## Ideas from the superseded draft (`archive/specs_scratch_draft.txt`), unranked
- τ ≥ 0.90 rank-1 lock with the rest hedged by same-formula isomers.
- Scaffold / NPClassifier-superclass consensus of the top library analogs as a ranking feature.
- PubChem tier as a third retrieval tier.
- Lean LightGBM (subformula coverage, neutral losses, ppm, polarity agreement) — must train on FPNet/ranker-unseen queries.
