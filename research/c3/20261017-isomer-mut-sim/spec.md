# Spec: isomer-mut-sim, stage I1 (reach, CPU)

Campaign `20261017-c3-iso`. Gate block: gate_report `### 2026-10-17: c3-iso campaign, PRE-REGISTERED`, bullet I1.
The human's idea: after fusion, the candidates are on the list because the query's peaks match them; a Class-3 truth
is likely an **isomer** of a retrieved candidate. Mutate the fused list (scaffold-aware, biogenetic edits), score
retrieved + mutants together with the forward simulator, push a confident mutant to rank 2 (and others to the tail).
This stage measures only whether the truth is **reachable** that way. I2 (ICEBERG, GPU) and I3 (replay) come later.

## Key constraint
Fused candidates already sit at the query mass, so edits must be **mass-preserving**: the mutant's molecular formula
equals the parent's exactly (checked with RDKit `CalcMolFormula`, after removing stereo). K1's mutator
(`delta_m_mutator.mutate_anchored`) makes mass-changing edits of shifted analogs; reuse its style (RDKit reaction-free
atom edits, sanitize, deadline) but not its operator set.

## Operators (`build/isomer_mut.py`, one parent → set of child SMILES, ≤ 2 moves)
Each operator must keep the formula identical; children failing sanitize or the formula check are dropped.
1. **Substituent shift**: detach a terminal group (OH, OMe, Me, Cl/Br/F, NH2, C(=O)O–, O-glycoside as a unit) from an
   aromatic or ring atom and reattach it to another ring atom of the same ring system carrying an H (regio-isomers).
2. **O-/C-methyl swap**: OMe ↔ CH2OH on the same attachment atom; O-Me on another free OH (methyl migration).
3. **Glycoside / ester position**: move a sugar or acyl group between free OH positions of the aglycone.
4. **Double-bond / ring-junction shift**: move a C=C one position in a non-aromatic chain or ring (keeps formula).
5. **Chain isomer**: n-alkyl ↔ iso-alkyl branch (move one CH3 along a saturated chain).
6. **Two-move composition** of 1–5 (cap the count per parent).
Biogenetic prior (optional, ordering only): prefer edits whose attachment atom is an O/N-bearing position typical of
natural products (O-methylation, glycosylation, prenylation positions). Record the operator per child.

## Pipeline (`build/isomer_reach.py` + tests)
1. Per molecule of each cohort, take the V22 fused top 10 (after `casmi_post.run` with the V22 cfg, line `c3`).
2. Mutate each parent (≤ 2 moves, cap 300 children per molecule, deadline 60 s per molecule, Pool ≤ 4 procs; reuse the
   task fan-out of `build/novel_p4_inputs.py`).
3. Dedupe by plain IK14 (RDKit InChIKey[:14] without tautomer canonicalisation; tautomer IK14 only for the final top 20),
   drop children that equal any retrieved candidate.
4. CPU fragment pre-screen: score each child against the query's **measured** MS/MS (R10: the reference is the
   query spectrum, never a truth-derived spectrum) using `build/frag_rescore.py` (`score_molecule`); keep the top 20
   per molecule. If query spectra are missing for a cohort in the dump, load them from the parquet the dump came from
   (pseudo_c3: `p1_novel/pseudo_c3.parquet` or train.parquet rows by molecule id); record where they came from.
5. Reach (truth only reports, never selects): reach_all = truth tautomer IK14 ∈ all children; reach_20 = truth ∈ top 20;
   rank of truth among the 20 when present.

## Output
`research/c3/20261017-isomer-mut-sim/i1_result.json` + `i1_result.md`: reach_all, reach_20 on pseudo_c3 (with a
Wilson 90% CI), splits acyclic/ring and timsTOF (15), children per molecule, operator that produced each hit, CPU s
per molecule, and the same counts on C1_npx/C1_xlib/C2/C2_hard (reach of the truth when the truth is NOT in the
fused top 10 is the interesting C1/C2 number; also how often a child of the truth's neighbours is the truth).
**Verdict: DEAD iff reach_20 < 0.10 on pseudo_c3.** Then STATUS → `needs-human` (checkpoint; I2 needs a GPU go) or `dead`.

## Tests (`build/test_isomer_mut.py`)
- every child of a set of real natural-product parents (from the pseudo_c3 dump, not toy SMILES) has the parent's
  formula; at least 1 child per operator on a parent where it applies; no child equals its parent IK14.
- known regio-isomer pair (e.g. 4-methoxy- vs 3-methoxy-cinnamic acid; quercetin 3-O- vs 7-O-glucoside) is reachable in 1 move.
- Non-vacuous: on 20 real pseudo_c3 molecules, mean children > 0.
