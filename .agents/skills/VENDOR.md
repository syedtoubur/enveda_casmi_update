# Vendored third-party skills (2026-10-18, conductor plan Phase 5)

Copied into both .claude/skills/ (Claude Code, opencode) and .agents/skills/ (Antigravity, Qoder). Reviewed before
vendoring: network calls go only to public scholarly APIs (arXiv, bioRxiv, Crossref, OpenAlex, NCBI, EBI, DataCite).

| skill | source | licence |
|---|---|---|
| literature-review, paper-lookup, citation-management, deepchem, torch-geometric, torchdrug, pytorch-lightning | github.com/K-Dense-AI/claude-scientific-skills @ a1043ef | MIT / Apache-2.0 (per SKILL.md) |
| nature-reader, nature-paper-card | github.com/yuan1z0825/nature-skills @ f8797fe | Apache-2.0 repo, MIT per SKILL.md |

Left out on purpose:
- literature-review/scripts/generate_schematic*.py and generate_pdf.py, because they call paid OpenRouter image
  models and pandoc.
- citation-management/scripts/search_google_scholar.py, because it scrapes Google Scholar.
- nature-*/evals, tests and agents.
- research-lookup was not vendored, because it needs the paid Parallel API.
