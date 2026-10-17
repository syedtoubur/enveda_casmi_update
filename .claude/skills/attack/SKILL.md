---
name: attack
description: Campaign step (tier H, headless): adversarial falsification of a CASMI spec before any code; writes attack.md with pre-registered death conditions. Invoked as `/attack research/<line>/<idea> --brief <path>`.
---

# Attack

You are the idea's adversary. Your win is a dead idea that would otherwise have reached the submit queue on a false premise. An idea that survives must have survived real attacks.

1. **Find the weakest joints**: for this specific mechanism, ask where the claim is false. Typical joints:
   - an empty or failed upstream stage (`GL_SCORES = {}`, no Engine 2 list, no PubChem list);
   - adducts outside the target set;
   - formula-group collisions;
   - the candidates the change displaces (the downside cohort, rank-1 displacements);
   - a gain that sits only on leak-hit molecules (BASE rank 1 = truth), which the LB will not reproduce (calibration.md);
   - the draft run not changing any molecule against the control's draft;
   - runtime.
2. **Write `attack.md`**: 3–6 rows of `attack | how stress_test.py checks it (data, fixture or harness subset) | idea dies if <numeric condition> | blocking yes/no`. Every check must be runnable on the harness dumps or synthetic fixtures, with no GPU (heavy checks run as a Kaggle CPU kernel, never on this device: CLAUDE.md §3). Mark an attack **blocking** when the idea must not ship unless the check actually ran: at the gate, a blocking attack left UNTESTED kills the idea.
3. Cite LB history only from scores on the same base pipeline (STATUS `lb_pipeline:` starts with the line's base notebook). An older pipeline's LB delta is an ASSUMED prior, not a death condition: the 10-12 promo-soften attack died on a pre-fusion LB delta.
4. Correct `spec.md` only where an attack exposes a factual error, and cite it. Bars stay as they are.

Done when attack.md has at least 3 rows, each with a numeric death condition, a concrete check and a blocking flag, and STATUS is `attacked`.
