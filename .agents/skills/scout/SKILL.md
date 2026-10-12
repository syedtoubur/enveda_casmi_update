---
name: scout
description: Gemini legwork before a CASMI campaign grill, or for one targeted question from /replan. Scans public Kaggle notebooks and discussions for this competition plus recent papers, and writes research/scout.md (one page). Use on /scout or /scout "<question>".
---

# Scout (Gemini, headless or interactive)

Argument: none (a general scout before a grill) or one quoted question (a targeted scout).

1. **Collect**:
   - public Kaggle notebooks for `enveda-CASMI26-molecule-id-mass-spectra` with a public LB above our best (`kaggle kernels list --competition <slug> --sort-by scoreDescending`);
   - competition discussions from the last 14 days;
   - arXiv / bioRxiv / OpenAlex papers on MS/MS structure elucidation (spectral retrieval, fingerprint prediction, forward simulation, reranking).

   For a notebook, use `python .agents/skills/analyze-notebook/scripts/compare.py <theirs> casmi-notebook-0-433.ipynb`.
2. **Filter**: keep an item only if you can name the mechanism and where it would act in our pipeline (`build/casmi_post.py` knob, or upstream: ranker, pool, simulator input). Drop items that `research/LESSONS.md` already covers, and cite the lesson ID instead.
3. **Write `research/scout.md`** (≤ 60 lines; for a targeted scout, append a dated section). One bullet per item: link | mechanism | where in our code | post-scoring or heavy-path | evidence (their LB delta, paper metric).

Done when every bullet has a working link and a "where in our code" entry. An unsupported item is dropped, never kept vague.
