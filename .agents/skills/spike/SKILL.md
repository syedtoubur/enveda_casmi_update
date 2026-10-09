---
name: spike
description: Throwaway experiment that tests an idea's core claim in about 10 CPU-minutes on cached data, before any spec is written. Replaces /prototype for ML ideas. Writes spike/ and spike_result.md.
---

# Spike

A spike is **throwaway code that answers one question with a number**, for example "does feature X separate the truth from decoys on the val cohort?" or "how many test molecules would this change touch?".

1. Write the question as one line in `spike_result.md`. Check the premise against `research/<line>/BASE.md` first: a spike that measures a gap the base does not have is wasted.
2. Write one script, `<idea>/spike/<name>.py`. It runs on **local CPU** over cached arrays and must finish in ≤ 10 min. Keep it under 150 lines, with no tests and no polish. It prints the numbers. No `assert ceiling >= 0.004`: the spike reports, it does not argue.
   - Data lives on Drive (`specs/Dataset_intro/drive_folder_structure.txt`). If the claim needs GPU scorer outputs (FPNet / ICEBERG / GLACIER) or Drive-only data, run it once on Colab: `python research/bin/colab_run.py <idea> spike/<name>.py --kind spike`. That counts against the smoke CU cap (0.5 CU). If it returns 7 (`LOGIN NEEDED`) or STATUS has `colab: off`, do not wait for a login: spike on local CPU data only and mark the ceiling UNMEASURED. In the script, check `os.path.exists('/content/drive/MyDrive')`; never call `drive.mount()`.
   - For any pretrained fingerprint model, include known-label sanity structures (L26).
3. Fill `spike_result.md`:
   - The numbers, each one labelled **MEASURED** (what data, what script line) or **ASSUMED** (where the guess comes from). Counts from local `test.parquet` are placeholder data (L04): say so.
   - The **arithmetic ceiling** = (molecules affected × best rank gain) / n (L13), with each factor labelled MEASURED or ASSUMED. Use one cohort definition throughout; do not mix counts of different cohorts.
   - The verdict:
     - **SIGNAL**: the rank gain was measured (val cohort, cached scores) and the ceiling is ≥ 0.004.
     - **UNMEASURED**: the ceiling is ≥ 0.004 only with an ASSUMED rank gain. It may go on to a spec, but the spec must say UNMEASURED.
     - **NO SIGNAL**: ceiling < 0.004 even when the effect is real.
4. Never stage data files (L28). Keep intermediate outputs inside the idea folder (L29).
