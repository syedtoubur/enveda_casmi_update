---
name: fetch
description: Campaign step (tier L, headless, web): answer the open research/requests/*.md, one specific fact or one model search each. Invoked as `/fetch research/requests --brief <path>`.
---

# Fetch

Each open request (`status: open`) asks one question that the repo can't answer: a paper detail, a model's licence, whether pretrained weights exist, a dataset's overlap.

For each open request:
1. Search the web for the answer, open the primary source, and quote the line that answers it.
2. For a **model search**, list each candidate model with:
   - link and licence;
   - whether the weights are freely and publicly downloadable (the Kaggle external-data rule; NIST-licensed weights are not);
   - its training data, and the overlap risk with MassSpecGym, NIST or our val cohorts (leakage);
   - its size, and whether it can run offline on a T4 within the 9 h budget.
3. Append `## Answer` with sources, and set the request's status line to `status: answered`. If you found nothing, set `status: not-found` and list what you searched.

Done when no request is left `status: open`. Edit only the request files.
