status: answered
from: /ideate, campaign 20261014-qoder-structural (candidate `msalign-rerank`, research/ideas/20261014-candidates.md)

# Question
1. Does MSAlign (KrzakalaPaul/MSAlign-NeurIPS2026, arXiv 2605.19752) ship trained projection-head weights, and under
   which licence? Which data were the heads trained on (MassSpecGym only, or NIST)?
2. Licences of its frozen encoders: DreaMS weights (pluskal-lab/DreaMS) and MolDeBERTa. Are both freely and publicly
   downloadable and allowed in a Kaggle competition notebook (offline dataset upload)?
3. Inference memory/time on a 16 GB T4 for ~40 candidates × 400 molecules (precomputed molecule embeddings).

# Why it matters
Reusing released heads beats training them. If no heads are released, the fallback is training the two MLP heads on
Kaggle (≤ 4 h, the paper reports < 2 min/epoch), which must avoid the FPNet/ranker training structures (L01) and the
val cohorts (L19). A non-free licence or NIST-trained heads kill the idea.

## Answer

**Short answer:** No head weights are released. The heads must be trained. The encoders disagree on licence: DreaMS is
MIT-repo with public Zenodo weights, but MolDeBERTa is CC BY-NC-ND 4.0 and forbids redistribution, which blocks an
offline Kaggle dataset upload as written. Question 3 is not answered by any source.

### Candidate 1: MSAlign projection heads
- Link: https://github.com/KrzakalaPaul/MSAlign-NeurIPS2026 · paper https://arxiv.org/abs/2605.19752
- Licence: repo code is MIT (LICENSE file, per the repo sidebar). README: "External datasets and pretrained models
  retain their own licenses."
- Weights public: **no head weights found.** The top-level listing has no weights file, and `model_zoo/` holds only the
  five competitor baselines ("This directory contains the five competitor implementations reported in the MSAlign
  paper"). The README describes checkpoints as produced by training: "Completed checkpoints are skipped when the command
  is rerun." The abstract releases "All datasets, splits, candidate sets, and a unified implementation of MSAlign and
  baselines" and does not mention weights. Result: the fallback (train the heads) applies.
- Training data: MassSpecGym. Paper §5.1: "MassSpecGym [10] provides 231,104 carefully curated and standardized pairs",
  split "using the chemical formula as a criterion". NIST is not mentioned. §5.1 also names SpectraVerse as a supervised
  dataset; I did not confirm which heads used it.
- Overlap risk: the heads train on MassSpecGym pairs. Any val-cohort molecule drawn from MassSpecGym would leak. This
  must be checked by InChIKey against our cohorts; the web cannot do it.
- Size and time: parameter count not stated. Paper (§3, Candidate-based training): "less than 2minutes/epochs on
  MassSpecGym as reported in Table 10". Hardware for that figure is not in the visible text. The README gives no runtime
  ("Estimated total runtime for a V100 is [TODO: add runtime]") and says the pipeline "require[s] a GPU".

### Candidate 2: DreaMS (spectrum encoder E_ms, 1024-dim output)
- Link: https://github.com/pluskal-lab/DreaMS · weights https://zenodo.org/records/10997887
- Licence: repo "MIT license" (page text). The page gives no separate licence for the weights.
- Weights public: yes, the repo lists "Weights of pre-trained models: https://zenodo.org/records/10997887". No upload
  restriction is stated.
- Training data: "unannotated spectra from our new GeMS (GNPS Experimental Mass Spectra) dataset", "mined from the MassIVE
  GNPS repository". MassSpecGym and NIST are not mentioned. Overlap with MassSpecGym is not established by the page, and
  it must be checked by spectrum ID.
- Size: not stated.

### Candidate 3: MolDeBERTa (molecule encoder E_mol)
- Link: https://huggingface.co/SaeedLab/MolDeBERTa-base-123M-mtr
- Licence: **`cc-by-nc-nd-4.0`**. The page says: "Downloading the model requires prior registration on Hugging Face and
  agreeing to the terms of use." and "By downloading this model, you agree not to distribute, publish or reproduce a copy
  of the model." It is also non-commercial with attribution.
- Weights public: gated, not open. The no-redistribution clause conflicts with uploading the weights as an offline
  Kaggle dataset. As written the upload is blocked. Running it at inference without an upload is blocked by the offline
  rule. The fallback (training heads) also needs MolDeBERTa embeddings. If embeddings are precomputed elsewhere and only
  the embeddings are uploaded, the clause may not reach them. That is a licence question for a human; I do not give a
  legal conclusion.
- Training data: "up to 123 million SMILES strings from PubChem". Pretraining on PubChem is a broad overlap source, but
  the encoder sees SMILES text, not spectra; the leakage concern is smaller than for the heads.
- Size: 88.2M parameters, F32 safetensors (≈0.35 GB, my arithmetic from 88.2M × 4 bytes).
- Encoder conflict: the MSAlign paper text (§3) says MolDeBERTa; its Table 1 lists ChemBERTa for the molecule encoder;
  Table 4 lists ChemBERTa-13M as an alternative. I used the text. Confirm before use.

### Question 3: T4 memory and time
- **Not answered.** No source gives RAM/VRAM or inference time for ~40 candidates × 400 molecules. The README runtime is a
  TODO, and the paper gives only a training epoch time. This needs a measurement on a T4.

### Kaggle rule
- Not verified. A search did not return the CASMI 2026 rules page. The Kaggle competition page is
  https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra; its rules section must be read before
  relying on any external-model claim.

### Sources
- https://github.com/KrzakalaPaul/MSAlign-NeurIPS2026
- https://arxiv.org/abs/2605.19752 · https://arxiv.org/html/2605.19752
- https://github.com/pluskal-lab/DreaMS · https://zenodo.org/records/10997887
- https://huggingface.co/SaeedLab/MolDeBERTa-base-123M-mtr
- https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra
