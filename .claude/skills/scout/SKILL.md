---
name: scout
description: Campaign step (tier L, headless, web): literature and code scout for MS/MS structure-identification methods that could transfer to the CASMI pipeline. Invoked as `/scout research/campaigns/<id> --brief <path>`.
---

# Scout

You collect facts; ideate (tier H) judges them. Read the brief first.

1. Read `research/harness/loss_map.md` to learn the weakest buckets, and the previous `research/scout/*.md` so you don't repeat a source.
2. Search the web for methods published or released in the last 3 years. Coverage, in this order:
   - the loss-map buckets (retrieval misses, re-ranking of in-pool candidates, Class-3 de novo);
   - spectrum-to-structure models (MIST, CSI:FingerID, DreaMS, MassSpecGym baselines, ICEBERG/GLACIER/forward simulators, de novo generators);
   - CASMI and MassSpecGym leaderboard write-ups;
   - rank fusion, calibration and reranking from IR and recommender systems.
3. Open each source you keep, at least the abstract and the method section. Mark how deep you read it: `read`, `skimmed` or `abstract only`.
4. Write `research/scout/<yyyymmdd>.md` with at least 10 new sources. For each:
   - title, link and year;
   - the method in 2 lines;
   - `plugs in at:` our stage or bucket;
   - for any shipped model: its licence, whether the weights are public, and its training data (NIST-only weights are not competition-legal);
   - your confidence.

Done when the file holds at least 10 sources, each with every field above. Leave ranking and opinions out.
