status: answered
from: /ideate, campaign 20261014-qoder-structural (candidate `gen-probe-rerank`, research/ideas/20261014-candidates.md)

# Question
1. Is there an open-licence, publicly downloadable, pretrained spectrum→structure generator usable offline on one T4
   that takes MS/MS (+ formula) and emits sampled structures: DiffMS (coleygroup/DiffMS), FlowMS (arXiv 2603.18397),
   MS-BART, or GLMR (AAAI 2026, arXiv 2511.06259)? For each: licence, training data (MassSpecGym / NPLIB1 / NIST),
   checkpoint size, and reported sampling time per spectrum.
2. Does GLMR release code + weights for its "generated structure as a re-ranking query" stage?
3. Our own Track-2 generator (specs_novel, DreaMS + SELFIES DFS, P1 PASS with ts_dec): where are its checkpoints, and
   which structures did it train on (overlap with C2 / C2_hard / C2_np truths would make any harness read leaked)?

# Why it matters
The candidate uses generated structures only as a similarity probe that re-weights already-retrieved candidates (never
injected; EXPERIMENTS graveyard rows "Generated candidates inside the RRF" and "bio2"). Reusing a released generator
beats training one; a generator trained on the val cohorts' structures makes the harness number unusable (L19).

## Answer

**Short answer:** One generator is publicly downloadable under MIT: DiffMS, with a 0.94 GB checkpoint archive. Its
fine-tuned checkpoint documents MassSpecGym fine-tuning, so leakage needs checking. MS-BART has public Figshare weights
but no licence. FlowMS and GLMR have no code or weights found. Question 3 needs the repo and is not answered here.

### Candidate 1: DiffMS (coleygroup/DiffMS)
- Link: https://github.com/coleygroup/DiffMS · weights https://zenodo.org/records/15122968
- Licence: MIT. Repo: "DiffMS is released under the MIT license." Zenodo record: "MIT License".
- Weights public: yes. Repo: "We provide checkpoints for the end-to-end finetuned DiffMS model as well as the pretrained
  encoder/decoder weights" (Zenodo). Files: `diffms_checkpoints.tar.gz` 939.6 MB; `msg_preprocessed.tar.gz` 780.2 MB.
- Training data: the repo documents pretraining on "combined" (fp2mol config), and fine-tuning with "set the dataset in
  config.yaml to 'msg' for MassSpecGym or 'canopus' for NPLIB1". The page does not say which set the released finetuned
  checkpoint used. Overlap risk: if it is the MassSpecGym fine-tune, any val-cohort truth in MassSpecGym leaks. Check by
  InChIKey. NIST not mentioned.
- Sampling time: not stated on the repo page.
- Offline on T4: not established. Size is known; memory and speed are not.

### Candidate 2: FlowMS (arXiv 2603.18397, Nie & Gao)
- Link: https://arxiv.org/abs/2603.18397 · https://arxiv.org/html/2603.18397v1
- Licence: none found. No repo or licence link appears on the arXiv page, OpenReview page, or author homepages.
- Weights public: **no code or weights found.** Peng Gao's publication page lists the paper with "only a [paper] link and
  no [code] link". Appendix B says the implementation "builds upon the publicly available codebases of DiffMS", which is
  not a release of FlowMS itself.
- Training data: decoder pretrained on "2.8M fingerprint-molecule pairs sampled from DSSTox". §3.3: "We exclude all
  molecules from the NPLIB1 and MassSpecGym from pretraining data." Fine-tuned on NPLIB1 for 50 epochs (App. B). NIST not
  mentioned. Encoder: MIST (secondary sources; not verified on the paper).
- Checkpoint size and sampling time: not stated.

### Candidate 3: MS-BART (OpenDFM/MS-BART, NeurIPS 2025)
- Link: https://github.com/OpenDFM/MS-BART · weights https://figshare.com/articles/dataset/MS-BART-Model-Weights-Data/30393544
- Licence: **none found.** The top-level listing has no LICENSE file and the sidebar shows no licence. Treat as all rights
  reserved until the authors confirm.
- Weights public: the README says "Download the preprocessed datasets and original model weights from Figshare", with
  "The final MS-BART model on CANOPUS dataset" and "The final MS-BART model on MassSpecGym dataset". No Figshare licence
  is stated. Alternative MIST encoder weights: https://zenodo.org/records/19685145.
- Training data: pretraining on "clean 4M pretrain dataset" from `MassSpecGym_molecules_MCES2_disjoint_with_test_fold_4M.tsv`.
  The name says disjoint from the MassSpecGym test fold, so train and val overlap are not excluded. Fine-tuned on
  MassSpecGym (`data/MassSpecGym/data/MassSpecGym.tsv`) and CANOPUS. NIST not mentioned.
- Checkpoint size and sampling time: not stated.

### Candidate 4: GLMR (arXiv 2511.06259, AAAI 2026)
- Link: https://arxiv.org/abs/2511.06259 · https://arxiv.org/html/2511.06259
- Licence: none found; no code exists to license (see Question 2).
- Weights public: not found.
- Training data: MassSpecGym training set. §4.1: "Our model is trained in two stages using spectrum-molecule pairs from
  the MassSpecGym training set." It also says "To avoid data leakage, we exclude molecules that appear in the MassSpecGym
  training set." MassRET-20k is built from NIST: Appendix A says "we construct MassRET-20k using the NIST2020 database,
  which is a non-open-source database with license restrictions". A NIST-derived set is a kill point for reuse.
- Sampling time: Appendix G: "The pre-retrieval stage of GLMR requires 0.41s" and "The generative retrieval stage of GLMR
  takes 0.69s", so 1.10 s per instance on MassSpecGym (Table A3). Parameter count is in Table A3 but was not extracted.

### Question 2: does GLMR release code and weights?
- **Not found.** A search found no GitHub or model link for GLMR. A ResearchGate listing links to
  https://github.com/HassounLab/JESTR1/, which is JESTR's code, not GLMR's. The paper text I read has no code-availability
  statement. This is "not found", not "confirmed not released". The authors' page is the next check.

### Question 3: our own Track-2 generator
- **Not answered from the web.** The checkpoints and training structures are repo-local (specs_novel), and the brief does
  not allow opening the repo. This needs a repo-side check of the checkpoint path and its training-set ID list against
  C2 / C2_hard / C2_np.

### Sources
- https://github.com/coleygroup/DiffMS · https://zenodo.org/records/15122968
- https://arxiv.org/abs/2603.18397 · https://arxiv.org/html/2603.18397v1
- https://github.com/OpenDFM/MS-BART · https://figshare.com/articles/dataset/MS-BART-Model-Weights-Data/30393544 · https://zenodo.org/records/19685145
- https://arxiv.org/abs/2511.06259 · https://arxiv.org/html/2511.06259
- https://www.catalyzex.com/author/Peng%20Gao · https://people.cs.vt.edu/penggao/publication.html
