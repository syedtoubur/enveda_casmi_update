---
name: ideate
description: Campaign step (tier H, headless): invent the next CASMI ideas from scout notes, the loss diagnosis, past near-misses and mutation operators; ranked, falsifiable, at least two genuinely new. Invoked as `/ideate research/campaigns/<id> --brief <path>`.
---

# Ideate

You are the loop's inventor. A variant of a past idea is cheap and usually noise (fusion beats tuning; LB noise ±0.004). Aim for mechanisms that change **which candidates** reach the top 25 or **what evidence** ranks them.

1. **Gather** (only from the brief's reads):
   - the newest scout file;
   - the diagnosis (its buckets and hypotheses);
   - calibration.md (which harness signal tracks the LB);
   - campaign history (what died, and why);
   - LESSONS IDs and the EXPERIMENTS dead-ends.
2. **Generate** at least 12 raw ideas with these operators:
   - **transfer**: a scout method aimed at a diagnosis bucket;
   - **combine**: two orthogonal sources fused (an RRF list, a re-rank feature);
   - **mutate** a near-miss: unbundle, invert, move the lever to another cohort or class;
   - **fill a gap**: an idea for a class or bucket that nothing targets yet (Class-3 de novo is the largest open gap; Class-2 new models);
   - **remove**: a stage whose cost exceeds its value.
3. **Kill**:
   - any idea a LESSON or a recorded dead end already refutes (cite it);
   - any that can't be measured on the scorecard cohorts or by a ceiling script;
   - any that breaks the Kaggle rules (offline, ≤ 9 h on a T4, only freely and publicly available weights).

   For a training idea, file `research/requests/<yyyymmdd>-model-<slug>.md` (`status: open`), asking whether an open-licence pretrained model already does it. Reusing a model beats training one. Kaggle training is allowed at ≤ 4 h per run from the weekly pool; a Colab run is a human request.
4. **Write** `research/ideas/<yyyymmdd>-candidates.md`, using the brief's columns for each idea and ranking by EV. At least 5 candidates; at least 2 tagged `novelty: new`, meaning not a parameter change of anything in the campaign history or EXPERIMENTS.
5. Set `state.json` `ideate` to null.

Done when the file holds ≥ 5 ranked candidates, each with a falsifier, ≥ 2 of them `novelty: new`, and `ideate` is null.
