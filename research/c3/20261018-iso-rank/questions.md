# Questions: 20261018-iso-rank

- 2026-10-18 09:10 (g4-fix round 1): G4 "FAIL" was a missing log download (Kaggle API 429), not a notebook failure (see
  g4_fix_1.md). I set STATUS `g4-timeout` and did not re-push, so `python research/bin/kaggle_push.py --check
  research/c3/20261018-iso-rank` re-finalizes once the rate limit clears. If you would rather re-push, the command is in
  g4_fix_brief.md.
