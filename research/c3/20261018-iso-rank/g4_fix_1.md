# G4 fix 1: 20261018-iso-rank

## Symptom
G4 FAIL with **every** required line and counter "not printed", and the brief's log tail reads `(no log downloaded)`.
`kaggle_output/` holds the CSVs and `gl_work/gl_out.json.meta.json` but **no `*.log`**.

## Root cause: log download failed (infra), not the notebook
- The kernel status is `complete` (watch.log t+1360s), so the notebook ran to the end.
- `gl_out.json.meta.json`: `"status": "ok", "device": "cuda", n_mols_scored 11/11, 755 cands, errors []`, so the GPU
  GLACIER stage ran cleanly on the draft (fast-commit) slice.
- The same `OUTPUT_FILES` pattern fetched a `*.log` for 14 other ideas, so the pattern is fine.
- Re-fetching just the log now (`kaggle kernels output syedtoubur/casmi26-iso-rank --file-pattern "\.log$"`) returns
  `429 Client Error: Too Many Requests ... ListKernelSessionOutput`. The finalize fetch most likely lost the log the same
  way: the CSVs came through and the log page did not.
- With an empty log_text, `check_log` reports every required marker as missing. That is a false FAIL.

## Fix
1. No code change. The notebook is not shown to be broken, so changing the patch would be unfounded. A re-push would spend
   another ~23 min of GPU on the same code, and its log fetch could hit the same 429.
2. Set STATUS to `g4-timeout` (kernel, push_id and kernel_version kept). `kaggle_push.py --check` (run by /daily-digest)
   finalizes g4-timeout ideas: it re-downloads the log once the rate limit clears and scans it for real.
3. If that finalize still shows missing markers **with a log present**, that is a real G4 FAIL, and the normal g4-fix
   round 2 follows.
