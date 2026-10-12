---
name: attack
description: Campaign step (Opus, headless): adversarial falsification of a CASMI spec before any code; writes attack.md with pre-registered death conditions. Invoked as `/attack research/exploit/<idea> --brief <path>`.
---

# Attack

You are the idea's adversary. Your win is a dead idea that would otherwise have wasted an LB slot. An idea that survives must have survived real attacks.

1. **Find the weakest joints**: for this specific mechanism, ask where the claim is false. Typical joints:
   - an empty or failed upstream stage (`GL_SCORES = {}`, no Engine 2 list, no PubChem list);
   - adducts outside the target set;
   - formula-group collisions;
   - the candidates the change displaces (the downside cohort);
   - the leak proxy flipping the sign;
   - the 12-molecule draft run firing no counter;
   - runtime.
2. **Write `attack.md`**: 3–6 rows of `attack | how stress_test.py checks it (data, fixture or harness subset) | idea dies if <numeric condition>`. Every check must be runnable by Sonnet on the harness dumps or synthetic fixtures, with no GPU.
3. Correct `spec.md` only where an attack exposes a factual error, and cite it. Bars stay as they are.

Done when attack.md has at least 3 rows, each with a numeric death condition and a concrete check, and STATUS is `attacked`.
