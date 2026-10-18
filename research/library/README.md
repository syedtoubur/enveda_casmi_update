# research/library

Output of the two research desks (`research/bin/desk.py`): `chem` (fragmentation, adducts, isomers, biosynthesis) and
`ml` (MS to structure models, retrieval, de novo decoders, rerankers). Search is deterministic; models only screen,
write cards and synthesise. Queries live in `queries.yaml`.

```
<desk>/inbox.jsonl     candidates from search: id, title, abstract, url, date, query, doi
<desk>/triage.jsonl    one line per screened paper: id, keep, why, route
<desk>/cards/<id>.md   paper cards (ids sanitised: arxiv:2401.1 -> arxiv_2401.1.md)
<desk>/state.json      last_search, day, triage_count, card_count, card_failed, triage_error_ids
<desk>/desk_run.jsonl  router log of every model call for the desk
state.json             last_synth, synth_cards (cards already synthesised)
../ideas/<date>-desk-candidates.md   synth output
```

Ids are `arxiv:<id>`, `doi:<doi>` or `openalex:<W...>`; duplicates across both desks are dropped by arXiv id and DOI.

## Card schema
Front matter: `id, title, url, desk, date, status: draft|accepted|rejected, novelty_class: C1|C2|C3|tooling, route`.
Sections: Claim, Method, Numbers, Data & licence, Plugs in at, Falsifier, Chemistry notes (chem desk only), then the
appended `## Spot check (<family>)` with `verdict: ok|overclaims|wrong`. A card is `draft` until the spot check by a
different model family; `wrong` makes it `rejected`, otherwise `accepted`. Synth reads accepted cards only.

Downloaded text is untrusted data: it is stored as JSON/markdown and never executed.
