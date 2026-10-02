# specs_novel — plan and constitution

*Version 3.1, 2026-09-28.*

v3.1 applies the P0-planning source audit (amendments A1–A9; details in §2 and §4 and in roadmap P0).
It changes no architecture decision. The DreaMS load path becomes a one-off conversion to a plain
`dreams_embed.pt`; stereo is stripped before SELFIES encoding; the mask rules cover skipped atoms
and effective bond orders; a DBE-prune violation disables the prune instead of failing G0.4.

v3.0 replaces v2.1, which was built on MADGEN (§0). The generator is now a **SELFIES decoder**
built from these parts:

- It is trained on top of frozen **DreaMS** spectrum embeddings.
- It uses the **MassSpecGym** (`MSGYM_novel/`) tokenizer, formula encoding and metrics.
- At test time, a **constrained DFS** searches its token space.

The module attaches to **Approach B**, the simple ICEBERG baseline (LB 0.289). Every number in this
file is either measured, with its source, or marked **to measure**. File:line references point into
`MSGYM_novel/` (pluskal-lab/MassSpecGym @ `f259fe3`) and `DreamS/` (pluskal-lab/DreaMS @ `dbec3a0`).
Background: `Starting Point/summary.md`.

---

## 0. History: why MADGEN was dropped (2026-09-28)

Reading the source of v2.1 (`MADGEN/`) showed that MADGEN is the wrong fit. The reasons are below;
the v2.1 text is in git history.

| MADGEN property | why it rules MADGEN out here |
|---|---|
| Input = a ring scaffold **plus** the true heavy-atom set | For Class 3 we would have to predict both. The published "predicted scaffold" numbers leak: the truth sits in the retrieval list, and the vote uses the true formula. |
| No scaffold means the molecule is dropped | It cannot generate **acyclic** structures (lipids, linear peptides, fatty acids). |
| Trained on at most 71 heavy atoms | Test masses reach 1,159 Da. |
| Its only input is 50 peaks | It has no adduct, CE or precursor input. |
| `sample.py` needs the true SMILES; the pinned stack is Python 3.9 | A full port would have been needed. |

`madgen_files_intro.txt` is kept as a record. Elsewhere MADGEN appears only as a published reference number (§1.2) and in the v2.1 → v3.0 comparison (§2.3).

---

## 1. Mission

### 1.1 The wall

Every pipeline so far retrieves from a closed candidate pool: COCONUT 2.0 ∪ train structures,
712,199 rows (summary §1). Two kinds of molecule therefore score 0 whatever the ranking:

- **Class 3.** The structure is not in PubChem at all (competition data page).
- **Class 2 outside COCONUT.** The structure is in PubChem but not in our pool.

`specs_novel` adds a **generator** that proposes structures outside the pool. A generated structure
scores if its InChIKey14 matches the truth's, whichever class the truth belongs to.

Expanding the pool with MassSpecGym's 1M/4M PubChem-derived molecule sets could reach out-of-pool
Class 2. It **cannot** reach Class 3, whose structures are by definition not in PubChem. It is
therefore a **separate side-track** with its own one-flag probe, and not part of this spec.

### 1.2 What it can be worth (stated before any measurement)

```
ΔLB ≈ f(C3) · MRR_novel(C3)  +  f(C2_out) · MRR_novel(C2_out)  −  displacement(C1, C2_in)
```

- **`f(·)`:** the class fractions of the hidden test. They are hidden and **unknown**.
- **`MRR_novel`:** measured offline on the pseudo-Class-3 cohort (roadmap P4). **The prior is
  low.** Published exact-match rates on MassSpecGym (`MSGYM_novel/results/de_novo*.csv`):

  | method | formula given | top-1 exact | top-10 exact |
  |---|---|---|---|
  | MassSpecGym SMILES / SELFIES transformer | no or yes | 0.00 | 0.00 |
  | Random chemical generation (formula-constrained) | yes | 0.00 | 0.00 |
  | MADGEN | yes | 0.0131 | 0.0154 |
  | DiffMS (best listed) | yes | 0.0230 | 0.0425 |

  MassSpecGym's test fold is MCES-disjoint from training, so it is harder than a random split. That
  still makes a few percent at top-25 the realistic ceiling even with the **true** formula. Our test
  is 100% timsTOF (summary §1), which no public model was trained on.
- **`displacement`:** the MRR lost when novel candidates push a correct pool answer down the list.
  It is measured on the C2-displace cohort (P4) and kept small by the v1 slot policy (§3.9).

The expected gain is small and uncertain. So every phase has a gate written down before its
measurement, and the project stops at the first gate that fails.

### 1.3 Decisions already taken

| decision | choice | why |
|---|---|---|
| Class 1 and Class 2 answers come from | **Approach B** (`build/simple_sim_submit.py`, LB 0.289) | ICEBERG is already wired in and verifies generated candidates. The one-flag control is clean (summary §5.6). |
| Generator | **Our own SELFIES decoder**, conditioned on DreaMS embeddings | A token-by-token decoder allows the formula-constrained DFS the user designed. It has no scaffold or ring requirement and no 71-atom limit. |
| Spectrum encoder | **DreaMS `embedding_model.ckpt`, frozen** (Drive `casmi_2026/DreamS_Embedding_Model`) | Pretrained self-supervised on GeMS. The train corpus embeddings are already cached (`_dreams_out/emb`, `casmi_dreams_colab.ipynb`), so training needs no DreaMS forward pass. |
| Decoder (chemical-language) pretraining | **Skipped** | DreaMS pretrains the encoder only (§2.2). Pretraining the decoder on MassSpecGym's 4M molecules is the **single pre-registered rescue** if P3 fails. |
| Tokens | **Word-level SELFIES** (`SelfiesTokenizer`) | Every string decodes to a molecule, and one atom token means one element, so count masks are exact (§4). |
| Memory | **One token per spectrum** (DreaMS vector + adduct + CE) **plus a formula token** | Handles 1–16 spectra per molecule and adds the adduct and CE conditioning that MassSpecGym's model lacks. |
| Formula at test | **Top-k from our formula stage** | Training uses the true formula. At test, the formula both conditions the decoder and supplies the DFS masks. |
| Stereo and isotopes (A9) | **Stripped** before SELFIES encoding (`RemoveStereochemistry`, isotope labels to 0) | ik14 ignores both, so they cannot score. Stripping shrinks the vocabulary and keeps one atom token = one element. The round-trip is judged on ik14. |
| Search strategy (P3) | **A 3-method bake-off at an equal candidate budget**: prefix-seeded batched sampling, plain masked beam (control), diverse masked beam. Winner selected on `dec_val`, G3 judged once on `pseudo_c3` | Batched generation keeps the gate cheap and the control arm shows whether diversity engineering earns anything. Seeding must be at depth `L_SEED` ≈ 4: at depth 1 the mask leaves only "which element", so there are no distinct roots to spread over (§4.3) |
| Where the rerank applies | **P4 only, to both arms.** G3 is decoder-only | G3's metrics (top-25 max Tanimoto, exact@25) are **set** metrics, so reordering the 25 cannot change them; and giving only our arm a reranker would measure the reranker, not the decoder |
| DreaMS on Kaggle (A2) | **Vendored and converted** | The real package loads the head once on Colab and saves `dreams_embed.pt` (plain `state_dict` + args dict). Kaggle runs the torch-only `build/dreams_infer.py`: no pip, no pickled `dreams.*` classes, no `ssl_model.ckpt`. |

---

## 2. What the two repositories provide (source-verified)

### 2.1 MSGYM_novel (MassSpecGym): a framework, not a generator

| fact | evidence |
|---|---|
| **No de novo weights** are shipped. The repo is benchmark infrastructure: data loaders, transforms, baselines and metrics. | tree listing; README "Train and evaluate your model" |
| `SmilesTransformer` is an `nn.Transformer` encoder over `(n_peaks, 2)` peak tokens, with a linear `src_encoder`. The precursor is prepended as a peak of intensity 1.1. | `models/de_novo/smiles_tranformer.py:50-60`; `data/transforms.py:60-97` (`SpecTokenizer`) |
| The formula is a 118-element count vector (H included) passed through an MLP and **added** to every encoder state. | `smiles_tranformer.py:104,171`; `data/transforms.py:270-298` (`MolToFormulaVector`) |
| Decoding is **greedy or multinomial only**. There is no beam, no masks and no constraint hook, so our search loop is our own (§4). | `smiles_tranformer.py:143-200` |
| The stock model **cannot take a DreaMS vector**: `forward` reads `batch["spec"]` peaks and derives its padding mask from `spec.sum(-1) == 0`. | `smiles_tranformer.py:78,86,137-138` |
| `SelfiesTokenizer` is a WordLevel vocabulary over SELFIES symbols with `<pad> <s> </s> <unk>`. With no `selfies_train` argument it **downloads** MassSpecGym from Hugging Face, so offline use must pass the alphabet. | `models/tokenizers.py:58-94`; `definitions.py:12-15` |
| (A4) `selfies_train` must be a **non-empty list of SELFIES** strings: `None` **or an empty list** triggers the HF download (`if not selfies_train`). A list is used as-is, so the alphabet comes **only** from that list; the union with `get_semantic_robust_alphabet()` must be made by us. `max_len` must be an int, because the base class calls `enable_truncation(max_len)`. | `models/tokenizers.py:73-81`, `:12-36` |
| `RandomDeNovo` builds random connected graphs that fit the formula, with valence assignment. It uses the **true** formula at test. It is our null baseline (P3). | `models/de_novo/random.py:181-240,1203-1204` |
| `DeNovoMassSpecGymModel` computes top-k exact match, MCES and Tanimoto. We subclass it to reuse the metrics. | `models/de_novo/base.py` |
| `setup.py` pins torch 2.3.0, numpy 1.24.4 and rdkit 2023.9.4, and requires Python ≥ 3.11. numpy 1.24 has **no Python 3.12 wheel**, so the package is installed `--no-deps` for training and **not at all** on Kaggle. | `setup.py:20-37,52` |
| (A5) Importing `massspecgym.data.transforms` runs `data/__init__.py`, which imports `datasets` and `data_module`: matchms, torch_geometric and pytorch_lightning come with it. `massspecgym.utils` imports pulp and picks an ILP solver **at import** (`pulp.listSolvers(onlyAvailable=True)[0]`, an `IndexError` if none). `simulation_utils/misc_utils.py` imports `distutils`, which Python 3.12 has only through the setuptools shim. | `data/__init__.py:1-2`; `utils.py:354`; `simulation_utils/misc_utils.py:3` |
| Code licence: MIT. The dataset on HF (`roman-bushuiev/MassSpecGym`) is MIT (memory: simulator-checkpoint-licence). | `LICENCE` |

### 2.2 DreaMS: a pretrained spectrum encoder

| fact | evidence |
|---|---|
| DreaMS is a transformer **encoder** over peaks plus a prepended precursor token. It is pretrained self-supervised (masked peaks and retention order) on GeMS. The embedding is the precursor token's output. | `dreams/models/dreams/dreams.py:138-185`; README |
| `embedding_model.ckpt` is a `ContrastiveHead` (a linear 1024→1024 layer on the precursor token) fine-tuned for spectral similarity. | `models/heads/heads.py:728-740`; `api.py:103-112` |
| **What that means for us:** DreaMS replaces *encoder* pretraining. It knows nothing about SMILES or SELFIES grammar, so **decoder** pretraining is a separate matter (§1.3). | by construction |
| Loading a fine-tuning head also loads `ssl_model.ckpt` as the backbone and downloads it if missing. That is internet at load time, so **P0 must provide or bypass it**. | `api.py:83-100` |
| (A1) Checkpoints download from **Hugging Face** `roman-bushuiev/DreaMS`. The Zenodo URL is commented out; Zenodo 10997887 is the archive only. | `utils/misc.py:48,64-72` |
| (A2) The head checkpoint's hparams hold only the head arguments. The **backbone architecture `args` live only in `ssl_model.ckpt`** (`DreaMS.__init__` calls `save_hyperparameters()`), so the head alone cannot rebuild the network. Both checkpoints pickle `dreams.*` objects (`DataFormatA`, `SpectrumPreprocessor`) and a `PosixPath`. Hence the one-off Colab conversion to `dreams_embed.pt` (§1.3). | `models/dreams/dreams.py:23-29`; `models/heads/heads.py:43-63`; `api.py:30-31` |
| Importing `dreams.api`, `heads.py` or `utils/data.py` pulls in wandb, pyopenms, umap, igraph, pynndescent, pandarallel, matchms and ase. Inference itself needs only `DreaMS.__init__`/`forward`, `models/dreams/layers.py`, `FourierFeatures`, `FeedForward` and the `DataFormatA` constants (`max_mz = 1000`, `max_tbxic_stdev = 1e-4`). `msml` is not imported by any `.py` under `dreams/`. | `dreams.py:1-18`; `utils/io.py:6-35`; `utils/data.py:1-46`; `setup.py:51` |
| Preprocessing: the 100 most intense peaks, m/z ascending, relative intensity, and a prepended precursor at 1.1. `normalize_mzs` is never set, so m/z above 1000 passes through unclipped (outside the pretraining range). | `utils/data.py:72-150`; `utils/spectra.py:165-174`; `casmi_dreams_colab.ipynb` cell 0 table |
| **Cached corpus:** the embeddings for `train.parquet` are on Drive at `_dreams_out/emb/emb_*.npy` (1024-d float32), with `corpus_meta.parquet`. The corpus is filtered as follows: test adducts only; enveda-180 capped at 150k and each other library at 400k (`random_state=0`); 0.2% intensity floor; peaks above precursor + 2 Da dropped. **CE is not stored**, so P0 re-derives it by replaying the deterministic filter. (A6) The cache was embedded with an **unpinned DreaMS HEAD** (`git clone --depth 1`, no commit recorded), so P0 checks the vendored model against the cache as well as Kaggle against Colab. | `casmi_dreams_colab.ipynb` cells 1, 3, 7, 8 |
| `setup.py` pins numpy 1.25.0 (no Python 3.12 wheel) and a git dependency (`msml`). DreaMS has **never been deployed on Kaggle**; summary.md has no DreaMS entry. Offline install is a P0 gate. | `DreamS/setup.py:27-52` |
| Code and weights: MIT (Zenodo 10997887). The weights are public, so they meet the "freely and publicly available" rule. What the contrastive head was fine-tuned on (MoNA and/or NIST20 pairs) is **to record in P0** from the checkpoint hparams, as a leakage note. (A3) The provisional answer from the training recipe is **MoNA, `[M+H]+` only, Murcko split** (`MoNA_A_Murcko_split_neighbours_[M+H]+_0.05Da.pkl`). A commented-out line mentions a NIST20+MoNA similarity table, so P0 confirms from the checkpoint. | README badges; `DreamS/setup.py:23`; `training/fine_tune.sh:45-71`; `training/train.py:223` |

### 2.3 v2.1 → v3.0: what changed

| topic | v2.1 (MADGEN) | v3.0 |
|---|---|---|
| Generation unit | whole graph per (formula, scaffold) | token by token, with a constrained DFS (§4) |
| Scaffold stage | required; recall was the kill point | **removed**; there is no scaffold input |
| Acyclic truths | unreachable | reachable |
| Size limit | 71 heavy atoms | `MAX_LEN` SELFIES tokens (P0 sets it from the corpus) |
| Spectrum input | first 50 peaks only | every spectrum of the molecule, via DreaMS, with its adduct and CE |
| Weights | a pretrained checkpoint | **we train** the decoder (Colab, P3) |
| Training exclusion | fixed by MassSpecGym | **ours to choose**: the pseudo-C3 structures and their scaffolds are held out |
| Main risk | scaffold recall | the exact-match rate of a from-scratch decoder (§1.2) |

---

## 3. Architecture

```
test molecule (1–16 spectra, 10 adducts, CE in eV)
 │
 ├─ Approach B pass 1, unchanged: centred neutral mass → pool window → lv, fr, ppm
 │
 ├─ ROUTER (§3.2) ── max lv ≥ τ_gen ─────────────────────────────► base list only
 ▼
 FORMULA STAGE (§3.3)     enumerate → filter → score → top FORMULA_TOPK
 ▼
 DreaMS EMBED (§3.4)      each spectrum → 1024-d (frozen embedding_model.ckpt)       ┐
 ▼                                                                                    │ GPU 1
 MEMORY (§3.5)            [proj(emb_i) + adduct_i + ce_i]_i  ⊕  [formula token]       │ worker
 ▼                                                                                    │
 CONSTRAINED DFS (§4)     SELFIES decoder, count masks, EOS gating, node budget        ┘
 ▼                        once per formula
 CONSTRAINT FILTER (§3.6) sanitize · single component · full formula incl. H · ik14 dedupe · in-pool flag
 ▼
 VERIFICATION (§3.7)      ICEBERG sv (GPU 0, existing worker) + MetFrag-lite fr
 ▼
 SLOT POLICY (§3.8)       merge the novel list into the base list
 ▼
 top 25, ik14-deduped
```

### 3.1 Base

`build/simple_sim_submit.py` runs as-is. For each molecule it produces:

- the candidate window;
- `lv`, `sv`, `fr` and `ppm`;
- the base order from `rank_two_tier(lv, sv, fr, ppm, τ = 0.75)`.

With `CFG.USE_NOVEL = False` the notebook must reproduce 0.289 (roadmap P6).

### 3.2 Router: a runtime trade-off, not a correctness gate

Generation only **adds** candidates, so a wrong "no generation" decision loses nothing that is already
in the base list. The rules are:

- Skip generation when `max lv ≥ τ_gen`. `τ_gen` starts at 0.75, the base's τ, and is tuned only by
  LB probe (P6).
- Negative-mode and rare adducts are **not** routed out. The decoder has an adduct input, and P4
  measures the hit rate per adduct. `GEN_ADDUCTS` restricts generation only if P4 shows an adduct is
  pure cost.
- Process molecules in ascending `max lv`, so the watchdog cuts the most library-like ones first.

`lv` does not depend on FPNet, so the `val_scripts` memorisation problem (summary §4.6) does not
apply. Pool-miss blindness does (memory: val-blind-to-pool-misses).

### 3.3 Formula stage

1. **Target mass.** Take the neutral mass as Approach B does: `simsim_common.neutral_mass1`, the
   median over spectra, centred by −1.45 ppm.
2. **Enumerate** formulas over C, H, N, O, P, S, F, Cl, Br, I within ±`PPM_WIN` (8.5 ppm).
3. **Filter.** Keep formulas whose RDBE is ≥ 0 and an integer and that satisfy the SENIOR rules.
   Element-ratio ranges are set in P2 from the training labels.
4. **Score.** Two terms, weighted as set in P2:
   - *subformula explanation*: the intensity fraction of fragments explainable within `TOL`;
   - *window prior*: how often the formula occurs among pool and library structures within
     ±`PPM_WIN`.
5. **Keep** the top `FORMULA_TOPK`. Also drop any formula whose elements are missing from the
   decoder's SELFIES vocabulary.

The formula has three uses:

- the formula memory token (§3.5);
- the count masks (§4);
- the full-formula leaf check (§3.6).

### 3.4 DreaMS embedding

- **Preprocessing.** Each spectrum of a routed molecule uses exactly the preprocessing behind the
  cache: `_dreams_peaks` from `casmi_dreams_colab.ipynb`, with the same floor, the same +2 Da cut and
  100 peaks.
- **Parity.** Any mismatch between the Kaggle and Colab preprocessing is silent quality loss. P0
  compares embeddings of the same spectra on both sides and requires cosine ≥ 0.999.
- **Placement.** DreaMS runs inside the novel worker on GPU 1. At test scale (1,213 visible spectra)
  it is cheap, but it is still **to measure**.

### 3.5 Memory

For each spectrum *i* of the molecule:

```
m_i = W_e · emb_i  +  E_adduct[adduct_i]  +  E_ce[bin(ce_i)]
```

- `W_e` is a linear projection from 1024 to `D_MODEL`. **`D_MODEL` = 512, `N_DEC_LAYERS` = 8,
  `N_HEADS` = 8, ff 2048, dropout 0.1** (fixed in P3; `gate_report.md` → G3).
- `ce_i` is the mean of `collision_energy_ev`, binned, with an "unknown" bin for missing values, so
  `E_ce` has **6 rows**: the 5 `CE_BINS` intervals plus "unknown". Training applies **CE dropout
  p = 0.3** to known CE, because `pseudo_c3` has CE on only 15 % of spectra while every test spectrum
  has one.
- `E_adduct` has 10 rows, the `simsim_common.TEST_ADDUCTS`.

One more token carries the formula:

```
m_f = MLP(formula_counts)
```

The formula MLP is the idea behind MassSpecGym's `MolToFormulaVector`. Here it is a separate token
instead of being added to every state.

The decoder (`nn.TransformerDecoder`, pre-norm) cross-attends over `{m_1 … m_k, m_f}` with no
positional encoding, because the spectra are a set. Only `W_e`, the embeddings, the formula MLP and
the decoder are trained; DreaMS stays frozen.

### 3.6 Constraint filter

A decoded structure is kept only if **all** of these hold:

1. the SELFIES decodes and `Chem.MolFromSmiles` sanitises it;
2. it is a single component (no `.`);
3. its **full** formula, including H, equals the target formula, so the mass is within `PPM_WIN`;
4. its ik14, after the metric's RDKit 2026.03.3 canonicalisation, is not already in the list.
   Repeats are counted (`freq`) before they are dropped.

Each survivor is then marked by pool membership:

- `in_pool`: already a base candidate, so it is dropped.
- otherwise: a genuine **novel** candidate.

The funnel counts are reported in P4: decoded, valid, single, formula, novel.

### 3.7 Verification

Novel candidates enter the **same** ICEBERG request path as pool candidates:

- the functions `build_requests`, `run_worker` and `spectrum_sv`;
- the query's adduct, the exact CE, the theoretical precursor and `MERGE_RULE = "max"`.

MetFrag-lite `fr` is computed exactly as for pool candidates. The decoder's own `logp` orders novel
candidates **among themselves** only.

Two limits carry over:

- fidelity is not discrimination (summary lesson 5);
- ICEBERG covers 6 of the 10 test adducts (summary §5.2).

### 3.8 Slot policy (versioned; each version is one LB probe)

| version | rule | risk |
|---|---|---|
| **v1 — tail** | Keep the base list's first `25 − K_NOVEL` entries. Fill the last `K_NOVEL` slots with novel candidates ordered by `sv` desc, then `logp` desc, then `fr` desc. | Only base answers below rank `25 − K_NOVEL` can be displaced. |
| v2 — arena | Within tier 2, a novel candidate goes above a pool candidate only if its `sv` exceeds that candidate's by δ. Tier 1 is never displaced. | Larger upside and larger displacement. δ comes from P4 and is confirmed only by the LB. |

---

## 4. Constrained DFS over SELFIES tokens

The user's design is kept: generate token by token, gate the end token, apply hard constraints
throughout, then verify. SELFIES makes it exact.

### 4.1 State per search node

For each node the search keeps:

- the token prefix;
- the decoder's KV cache;
- the cumulative `logp`;
- the **remaining heavy-atom counts** `r[e] = formula[e] − used[e]`;
- the **unsaturation used** `u` (rings plus π-bond increments);
- the **SELFIES grammar state**: current-atom valence, and how many upcoming tokens are index
  tokens because of a `[Branch*]` or `[Ring*]`.
- (A7) whether the next atom symbol would actually be **placed**. SELFIES skips an atom symbol when
  the derivation state has no free valence, and it lowers a requested bond order to what the
  valences allow. Counts and unsaturation follow the atoms placed and the **effective** bond orders,
  not the symbols read.

### 4.2 Hard masks, applied to the logits before any expansion

| rule | effect |
|---|---|
| An atom token of element `e` is legal only if `r[e] > 0` **and** it is read as an atom in the current grammar state **and** it would be placed (A7) | The heavy-atom formula can never be exceeded. Index tokens do not consume counts. A symbol SELFIES would skip is masked: canonical truth encodings never contain one, so the rule is sound and prunes dead tokens. |
| `</s>` is legal only if `r[e] = 0` for every `e` | The end token is reached only with the exact heavy-atom formula. |
| Length ≤ `MAX_LEN` | Bounded depth |
| Look-ahead: `Σ_e r[e] ≤ MAX_LEN − len` | Prunes prefixes that can no longer place their remaining atoms |
| **DBE prune (conditional):** `u ≤ DBE(formula)` | This is only **sound** when the formula has no S or P, because hypervalent S=O and P=O add π bonds without changing the formula's DBE. For formulas with S or P, the check moves to the leaf. (A8) Charged atoms (`[N+1]`, `[O-1]`, quaternary N) can also break the rings + π = DBE identity. |

**Soundness is tested, not assumed.** P0 decodes every true molecule in the corpus token by token
under its own formula. Any mask that would ever remove a truth's next token is a bug. The gate is
**0 violations** for the count, end-token and length rules.

(A8, pre-registered) The DBE prune is an optional speed-up. If the soundness run finds any DBE
violation, `DBE_PRUNE` is set to False and the check stays at the leaf only. That does **not** fail
G0.4. Count, end-token or length violations do.

**The grammar caveat is why the tracker is tested.** SELFIES reads the tokens after `[Branch*]` and
`[Ring*]` as indices, and it quietly lowers bond orders when valence runs out. The *reference* tracker
decodes the prefix with `selfies.decoder` and counts atoms and unsaturation with RDKit. The *fast*
incremental tracker used in search must match the reference on 10k molecules (P0 gate: 100%).

### 4.3 Search

**What the metric rewards.** G3 and G4.0 score **top-25 max Tanimoto** and **exact@25** — a *max* and
a *membership* over a set of 25. They are **order-invariant**: reordering the 25 cannot change either
number. So the search's job is to make the 25 *cover* chemical space, and ranking is a P4 concern
(MRR@25, top-1, top-10). Twenty-five isomers of one wrong core all score ~0.25; twenty-five distinct
scaffold families give a real chance that one matches.

**P3 runs three searches at an equal candidate budget of N = 256 sequences per molecule**, so GPU
compute is matched. All three drive `Tracker.legal_mask` / `push`; none reimplements the grammar.

| id | method | parameters |
|---|---|---|
| **A** | prefix-seeded batched sampling | masked expansion to depth `L_SEED` = 4 → top `K_ROOTS` = 8 distinct prefixes by `logp` → `M_ROLLOUT` = 32 sampled rollouts each; `T` = 0.7, top-p 0.90 |
| **B** | plain masked beam (**control**) | `BEAM` = 256, no diversity mechanism |
| **C** | diverse (grouped) masked beam | `BEAM` = 256, cross-beam penalty `ALPHA_DIVERSE` = 0.6 on ring and branch tokens; the **unpenalised** `logp` is kept for P4 ordering |

**Seeding must be deep, not at depth 1.** At the first position the derivation state is 0, so the mask
admits only atom tokens (`is_branch` needs `state > 1`, `is_ring` needs `state > 0`), and
`next_atom_state(bond, cap, 0)` returns order 0 for every bond prefix — `[C]`, `[=C]` and `[#C]` are
*degenerate* there. The distinct roots available at depth 1 are therefore just the elements in the
formula, 2–4 in practice. Scaffold topology is decided over the first ~4 tokens, which is where A
seeds.

**The 25 are then chosen from the ~256 survivors** by `maxmin` (take the highest-`logp` candidate, then
greedily add whichever candidate maximises the minimum Tanimoto distance to those already chosen;
Morgan r2, 2048 bits), with plain `logp` reported beside it.

**Best-first is deferred to P4, not rejected.** It is *not* inherently unbatched — the top-`B` frontier
can be popped and expanded in one forward pass. The reason it is not in P3 is that at this model size
the bottleneck is not the GPU but `Tracker.push` + `legal_mask` at 19 µs/step in Python (B = 256 ×
L = 171 ≈ 0.8 s per molecule of pure Python). More *samples* is cheap; more sequential *depth* is not.
P4 compares it against the P3 winner at an equal time budget.

Leaves then go through §3.6 and §3.7.

Hydrogens stay implicit throughout. SELFIES fills valences, so the H count is fixed only at the leaf,
which is why the §3.6 full-formula check does real work.

---

## 5. Contracts

1. **Licences.** MassSpecGym code and data: MIT. DreaMS code and weights: MIT (Zenodo 10997887).
   ICEBERG `msg_all`: MIT (memory: simulator-checkpoint-licence). Only freely and publicly available
   weights are used.
2. **Pinned artefacts.** Record the SHA1 of `embedding_model.ckpt`, `decoder.pt`, `selfies_vocab.json`
   and the ICEBERG checkpoints, and assert each at load.
3. **Exclusion sets.** Three learned components see structures:

   | component | what it was trained on | exclusion |
   |---|---|---|
   | Decoder (ours) | our training split | the pseudo-C3 structures **and their Murcko scaffolds** are held out (P1) |
   | ICEBERG `msg_all` | all folds of MassSpecGym | `msg_ik14` from `MassSpecGym1.5.tsv`: pseudo-C3 truths ∉ `msg_ik14` |
   | DreaMS contrastive head | spectral-similarity pairs (P0 records the source) | Reported as a caveat. It learned spectrum similarity, not structure identity. |

   Every offline number that claims to measure generalisation is computed on truths that satisfy
   all three, **and** are removed from the pool and the library (summary lesson 3).
4. **One change per LB submission, with a control** (summary lesson 1). The twins differ only in
   `CFG.USE_NOVEL`, and a test enforces this.
5. **One source of truth for the model.** `build/novel_decoder.py` defines the model, the vocabulary
   mapping and the tracker. Colab training and Kaggle inference both import it; neither
   reimplements it.
6. **Notebooks are generated.** They come from `build/mk_*.py`, and `simple_sim_submit` is embedded
   verbatim.
7. **Fail safe.**
   - The run mode defaults to a full run.
   - The watchdog drops generation before simulation.
   - A worker failure reverts the molecule to its base list.
8. **Output.** ik14 dedupe, at most 25 entries, no nulls.
9. **Evaluation.** Offline numbers are relative guides and the LB is the arbiter. `val_scripts`
   cannot see Class 3; only pseudo-C3 can, and only as an upper bound (timsTOF shift, §6).

---

## 6. Failure modes

| failure | effect | detection | mitigation |
|---|---|---|---|
| Exact-match rate near zero | No MRR gain (the published prior, §1.2) | P4 with the oracle formula | Kill at P4. Rescue order: 4M decoder pretraining (once), then per-peak memory (P7). |
| Formula miss | The truth is unreachable by construction | P2 top-k recall | Raise `FORMULA_TOPK`; add the window prior |
| DreaMS will not install offline | No embeddings on Kaggle | P0 install smoke on the Kaggle Python | `--no-deps` against Kaggle's preinstalled torch and Lightning, or vendor the model code and load the `state_dict` directly. `ssl_model.ckpt` is shipped or bypassed. |
| Preprocessing mismatch between cache and Kaggle | Silent quality loss | P0 cosine parity | One shared `_dreams_peaks` in `build/` |
| CE missing from the cache | No CE conditioning, or misaligned rows | P0 row-replay assertion (ik14 and precursor, row by row) | Re-embed the corpus with CE stored (Colab, **to measure** time) |
| m/z above 1000 | DreaMS is out of range | Flag `oob_mass` | Keep, and report MRR split by the flag |
| timsTOF domain shift | Neither DreaMS nor the decoder has much timsTOF | Pseudo-C3 on enveda-np-examples vs other libraries | Report both; the non-timsTOF number is an upper bound |
| Negative-mode or rare adducts | Few training examples | P4 per-adduct hit rate | `GEN_ADDUCTS` restriction only on evidence |
| SELFIES longer than `MAX_LEN` | Large molecules unreachable | P0 length distribution | Set `MAX_LEN` at the corpus p99.9; report the excluded share |
| Beam / scaffold collapse | All 25 candidates share one wrong core, so max Tanimoto stays near the random baseline | The P3 bake-off's **control arm B** (plain masked beam) against arms A and C; and the recorded count of unique ik14 per molecule | Prefix-seeded diversity (A), cross-beam penalty (C), `maxmin` selection over `logp` |
| Too few formula-valid leaves | Fewer than 25 candidates, so the selector is inert and the gate silently measures the funnel instead of the search | Recorded median unique formula-valid candidates per molecule (G3, recorded-not-gating) | Raise N; report the funnel beside every hit rate |
| Unsound mask | The truth is pruned silently | P0 soundness test (0 violations) | The DBE prune applies only to formulas without S or P |
| Tracker disagrees with SELFIES | Wrong counts, lost truths | P0 parity test | Fall back to the reference tracker (slower) |
| Displacement | Correct pool answers lose rank | P4 C2-displace cohort | v1 tail policy; small `K_NOVEL` |
| Runtime | Overruns 9 h | P4 per-molecule time; P5 projection | `NODE_BUDGET`, `FORMULA_TOPK`; GPU 1; ascending `max lv`; watchdog |
| Generated structure already in the pool | Wastes a slot | `in_pool` flag | Drop it |
| Worker crash or OOM | Lost molecules | Return code | Revert to the base list |
| Leakage in offline numbers | Pseudo-C3 MRR inflated | Assertions in the cohort builder | Contract §5.3 |
