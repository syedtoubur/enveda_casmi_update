# Glossary (plain language)

Used by the `/campaign-brief` skill. Sources: `CLAUDE.md`, `research/README.md`, `research/harness/gate_report.md`,
`research/LESSONS.md`. Where a definition is not written down in those files it says `TODO(human)`.

## The task and the score
- **MRR@25**: mean reciprocal rank of the true molecule in our top 25 guesses (rank 1 scores 1, rank 2 scores 0.5, absent scores 0). Higher is better; the competition metric.
- **InChIKey14**: the first 14 characters of a molecule's InChIKey, which encode only the 2D atom-and-bond graph. The metric compares molecules on this key (after tautomer canonicalisation), so stereochemistry and tautomers do not count.
- **Class 1**: the molecule has reference spectra in a library, so it can be matched by spectral similarity.
- **Class 2**: the molecule is in a chemical database (PubChem, COCONUT) but has no reference spectra, so we retrieve candidates and rank them.
- **Class 3**: a completely novel molecule that no database holds, so it must be generated; pure database retrieval cannot reach it.
- **Cohort**: a fixed set of validation molecules used to replay an idea offline (see the cohort names below).
- **C2**: 400 validation molecules where every library row of the truth is banned, simulating Class 2.
- **C1_npx**: 245 timsTOF validation molecules where only the query's own rows are banned, simulating Class 1.
- **C2_hard**: a 200-molecule Class-2 cohort. TODO(human): what makes it "hard".
- **C1_xlib**: a 300-molecule Class-1 cohort. TODO(human): how the library is excluded.
- **pseudo_c3**: a 600-molecule stand-in for Class 3, used to measure generators and Class-3 ideas. TODO(human): exact construction.
- **C2_np**: an advisory cohort (natural products); it failed its own base bar, so it is reported but never vetoes an idea.
- **Leak-hit vs other**: a molecule is a leak-hit when the base system already ranks the truth first (the base saw it in training). "Other" molecules are the leak-free rest. Read the "other" delta first; a gain that lives only on leak-hit molecules is not trusted (L40, L45).

## Measurement and decisions
- **Harness**: the offline replay system (`research/harness/`) that re-scores saved pipeline state with a changed setting, so an idea is measured without a new Kaggle run.
- **Scorecard**: one harness run that reports, for every cohort, the paired change in MRR@25 with a 90% confidence interval, the leak-hit/other split, and how many rank-1 truths were pushed down.
- **Gate**: the pre-registered rule that decides whether an idea is dead or goes forward. The 2026-10-13 gate is rank-only: an idea dies only on no gain on its target cohort, real harm on another, more than 1% of rank-1 truths displaced, or a failed attack.
- **G4**: the draft-run check on Kaggle: the probe kernel must finish, its change counters must be above 0, and its draft output must differ from the control's draft.
- **Paired control**: a replicate of the current base pushed in the same window as a probe, so the probe is compared with our own account's score, never an author's headline (L06).
- **LB noise floor ±0.004**: the same code scores within about 0.004 on the leaderboard from run to run, so smaller differences are noise (L05).
- **keep_bar**: the leaderboard score a probe must reach to be kept: paired control + 0.004, with its counters proving the change fired.

## Pipeline parts
- **FPNet**: the neural net that predicts molecular-fingerprint bits from a spectrum; its outputs feed the candidate ranker. TODO(human): confirm wording.
- **ICE / ICEBERG**: a spectrum simulator trained on MassSpecGym that predicts the spectrum of a candidate so it can be compared with the observed one (GLACIER is its sibling). Its training data overlaps the validation cohorts, so val gains from it may be leaked (L19).
- **RRF fusion**: reciprocal-rank fusion, a way of merging several ranked candidate lists into one; combining an orthogonal source this way is the lever that has paid before.
- **V22**: the current line base for the c3 and antigravity lines, kernel `casmi26-v22-rel150-e2r2-clean` (LB about 0.447-0.451).
- **E006 (v1)**: the earlier exploit base notebook, control LB 0.440 (since 2026-10-12).

## Idea STATUS names (first line of an idea's `STATUS` file)
- `proposed`, `specced`, `preflight-ok`, `attacked`, `harness-done`, `build`: the idea moves from written, to specced, to attacked, to measured on the harness, to being built into a probe notebook.
- `g4-pushed`: the probe is on Kaggle and being checked.
- `g4-fix`: G4 failed and an agent is fixing and re-pushing; after two failed rounds it becomes `needs-human`.
- `g4-timeout`: no G4 verdict after 60 minutes; `kaggle_push.py --check` finalises it later.
- `ready-to-submit`: passed G4; waiting for the human to press Submit.
- `submitted`, `scored-unpaired`, `keep-pending`: submitted and awaiting a leaderboard score or its paired control.
- `keep` / `revert`: the leaderboard verdict from `poll_lb.decide()`.
- `needs-human`: the loop is blocked on a human decision (see the idea's `questions.md`).
- `paused-quota`: paused because a model quota ran out.
- `dead`: killed by the gate or an attack; the `reason:` line says why.
- `instrument-ready`: an instrument idea (data dumps, not a score change) finished; it is never submitted.
