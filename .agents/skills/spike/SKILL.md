---
name: spike
description: Throwaway experiment that tests an idea's core claim in about 10 CPU-minutes on cached data, before any spec is written. Replaces /prototype for ML ideas. Writes spike/ and spike_result.md.
---

# Spike

A spike is **throwaway code that answers one question with a number**, for example "does feature X separate the truth from decoys on the val cohort?" or "how many test molecules would this change touch?".

1. Write the question as one line in `spike_result.md`.
2. Write one script, `<idea>/spike/<name>.py`. It runs on **local CPU** over cached arrays and must finish in ≤ 10 min. Keep it under 150 lines, with no tests and no polish. It prints the numbers.
   - Data lives on Drive (`specs/Dataset_intro/drive_folder_structure.txt`). If the claim needs GPU scorer outputs (FPNet / ICEBERG / GLACIER) or Drive-only data, run it once on Colab: `python research/bin/colab_run.py <idea> spike/<name>.py --kind spike`. That counts against the smoke CU cap (0.5 CU). In the script, check `os.path.exists('/content/drive/MyDrive')`; never call `drive.mount()`.
   - For any pretrained fingerprint model, include known-label sanity structures (L26).
3. Fill `spike_result.md`: the numbers, a verdict of SIGNAL or NO SIGNAL, and the **arithmetic ceiling** = (molecules affected × best rank gain) / n (L13). A ceiling < 0.004 is NO SIGNAL even when the effect is real.
4. Never stage data files (L28). Keep intermediate outputs inside the idea folder (L29).
