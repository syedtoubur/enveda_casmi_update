# Session Handoff Dossier
**Generated:** 2026-10-11 05:05:00  
**Source Conversation ID:** 33f5b233-feb1-4804-901a-af8ed7017f63  
**Active Objective:** Cross-Instrument Library Matching Statistical Evaluation & Smart Gate Integration for CASMI 2026 SOTA  

---

## 1. High-Level Mission & Active Objective
- **Competition**: Enveda CASMI 2026 (Predict 2D molecular structures from LC-MS/MS mass spectra, evaluated on tautomer-canonical InChIKey14 MRR@25).
- **Target Notebook**: [`casmi26-e006-window-40-ice-fallback-true.ipynb`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/casmi26-e006-window-40-ice-fallback-true.ipynb) (Public LB: ~0.440).
- **Objective Completed**: **Phase 3: Cross-Instrument Library Matching Statistical Evaluation with Exact Spectral Entropy & Precursor Window-Max Simulation on Kaggle**.
- **Next Frontier**: Integrate the empirically validated 4-Tier Smart Library Gate into `casmi26-e006-window-40-ice-fallback-true.ipynb` and verify fast-commit + offline constraints.

---

## 2. Settled Decisions & Ground Truth Findings

1. **Extreme-Value Decoy Inflation is Massive for Unfiltered Metrics**:
   - Single-decoy pairwise FPR at $\tau = 0.50$ is only **0.67%**.
   - In a realistic $\pm 10\text{ ppm}$ precursor window containing 50–200 candidates, the **Window-Max Raw Cosine False Lock Rate escalates to 10.83%** (and **16.83%** at $\tau = 0.40$).
   - Lowering raw cosine threshold without structural filters would severely pollute Class 2/3 queries.

2. **Spectral Entropy Has a Drastically Lower Decoy Tail**:
   - Spectral Entropy similarity ($1 - \frac{2 S(M) - S(P) - S(Q)}{\ln 4}$) suppresses window decoy false locks by **$3.8\times$** compared to raw cosine at $\tau = 0.50$ (2.83% FLR vs 10.83% FLR).

3. **The Smart Gate Completely Neutralizes Window False Locks**:
   - Precursor masking ($|m/z - M_{\text{prec}}| > 1.5\text{ Da}$), $N_{\text{matched}} \ge 4$, and high-$m/z$ fragment presence ($> 0.5 \times M_{\text{prec}}$):
     - At $\tau = 0.70$: Smart FLR = **0.00%** (0 false locks out of 1,200 negative queries, each with ~60 decoys).
     - At $\tau = 0.65$: Smart FLR = **0.08%** (1 out of 1,200), Smart Precision = **99.12%**, True Match Recall = **53.33%**.
     - At $\tau = 0.40$: Smart FLR = **0.58%**, Smart Precision = **95.61%**, True Match Recall = **72.00%**.

4. **Window Rank 1 Separation**:
   - When an authentic reference exists in Domain B, Spectral Entropy places the true match at **Rank 1 above all window decoys 91.17% of the time**.

5. **The 4-Tier Smart Library Gate Architecture (Empirically Validated)**:
   - **Tier 1 (`cos_masked >= 0.85` or `entropy >= 0.80`)**: Intra-instrument lock $\rightarrow$ Rank 1 pinned, 0 PubChem insertions (`untouched`), forward simulators bypassed ($\lambda = 0.0$).
   - **Tier 2 (`0.65 <= cos_masked < 0.85` or `0.65 <= entropy < 0.80`, $N \ge 2$)**: High cross-instrument $\rightarrow$ Rank 1 retained, aggressive PubChem suppressed, simulators dampened ($\lambda = 0.15$).
   - **Tier 3 (`0.40 <= cos_masked < 0.65`, $N \ge 4$, $\max(m/z) > 0.5 \times M_{\text{prec}}$)**: Rescued cross-instrument match (72.0% recall, 0.58% decoy window FLR) $\rightarrow$ Candidate in Top 2, gentle PubChem fallback only (slots `[15, 18, 21]`), $\lambda = 0.25$.
   - **Tier 4 (`cos_masked < 0.40` or fails filters)**: Class 2/3 query $\rightarrow$ Standard PubChem and simulator reranking.

---

## 3. Artifact Ledger
- [`c:\Users\h\OneDrive\Desktop\ENVEDA_casmi\.scratch\handoffs\smart_gate.md`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/.scratch/handoffs/smart_gate.md) - Master technical specification updated with Phase 1, Phase 2, and Phase 3 empirical tables.
- [`c:\Users\h\OneDrive\Desktop\ENVEDA_casmi\.scratch\build_eval_nb_v3.py`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/.scratch/build_eval_nb_v3.py) - Phase 3 notebook generator script.
- [`c:\Users\h\OneDrive\Desktop\ENVEDA_casmi\.scratch\kaggle_libmax_eval\casmi_cross_instrument_libmax_eval.ipynb`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/.scratch/kaggle_libmax_eval/casmi_cross_instrument_libmax_eval.ipynb) - Kaggle evaluation notebook (Version 4).
- [`c:\Users\h\OneDrive\Desktop\ENVEDA_casmi\.scratch\kaggle_libmax_eval\output_v3\phase3_entropy_window_max_report.json`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/.scratch/kaggle_libmax_eval/output_v3/phase3_entropy_window_max_report.json) - Complete Phase 3 empirical metrics JSON.
- [`c:\Users\h\OneDrive\Desktop\ENVEDA_casmi\.scratch\kaggle_libmax_eval\output_v3\phase3_quantile_distribution.csv`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/.scratch/kaggle_libmax_eval/output_v3/phase3_quantile_distribution.csv) - Table 1 quantile CSV.
- [`c:\Users\h\OneDrive\Desktop\ENVEDA_casmi\.scratch\kaggle_libmax_eval\output_v3\phase3_entropy_operating_curves.csv`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/.scratch/kaggle_libmax_eval/output_v3/phase3_entropy_operating_curves.csv) - Table 2 operating curves CSV.
- [`c:\Users\h\OneDrive\Desktop\ENVEDA_casmi\.scratch\kaggle_libmax_eval\output_v3\phase3_entropy_window_max_curves.png`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/.scratch/kaggle_libmax_eval/output_v3/phase3_entropy_window_max_curves.png) - Publication-grade density and ROC figure.

---

## 4. The Immediate Frontier (Next Steps)
1. **Apply Smart Gate to Production Notebook (`casmi26-e006-window-40-ice-fallback-true.ipynb`)**:
   - Implement Steps 1–5 from [`smart_gate.md`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/.scratch/handoffs/smart_gate.md):
     - Step 1: Add `evaluate_smart_gate_for_molecule(L, spectra, target, prec_mz)` in Cell 16.
     - Step 2: Store decisions `SMART_GATES[mid]` in Cell 18.
     - Step 3: Upgrade simulator candidate filter in Cell 20.
     - Step 4: Upgrade PubChem merge bouncer in Cell 22.
     - Step 5: Prevent friendly-fire demotion in Cell 30 (`rerank_row`).
2. **Execute Local Smoke Test / Dry Run**:
   - Verify notebook syntax, fast commit integrity (`SMOKE_N = 12`), and format compliance.
