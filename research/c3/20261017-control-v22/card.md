# Control: own-account V22 replicate (paired control for the c3 line, L05/L06)

- Notebook: `E:/casmi/lines/c3/scratch/baseline/casmi26-v22-rel150-e2r2-clean.ipynb`, unmodified.
- Pushed in the same window as the probe `research/c3/20261017-gen-rank1-keep` (V22 + GEN1_KEEP).
- Same Kaggle image as the probe (kaggle_push pins the line's `control:` image).
- Its LB score is the paired control: probe KEEP iff probe >= this score + 0.004 (`poll_lb.decide()`).
- Note: `lines.c3.control` in config.yaml still points at the E006 folder (placeholder); the human repoints it to this
  folder once this replicate has a G4 PASS and an LB score.
