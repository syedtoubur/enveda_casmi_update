# specs_novel — tech stack, artefacts, interfaces, budget

*Version 3.1, 2026-09-28 (P0 source-audit amendments A1–A9, plan.md). This version uses a SELFIES decoder on frozen DreaMS embeddings with a
constrained DFS (plan §3 and §4). MADGEN is dropped (plan §0). A value marked **to measure** has not
been measured yet; P0 and P4 fill these in. Measured values cite their source.*

---

## 1. Environment

| component | version | where | note |
|---|---|---|---|
| Python | 3.12 | Kaggle, Colab | as for Approach B |
| RDKit | 2026.03.3 (`casmi26-offline-rdkit-2026033`) | main kernel and novel worker | the metric's version; all ik14s are computed with it |
| Approach B base | `build/simple_sim_submit.py` | main kernel | embedded verbatim |
| ICEBERG | `coleygroup/ms-pred` @ `708148c2`, checkpoint `msg_all` | own venv, subprocess, **GPU 0** | SHA1s gen `c626bae1…`, inten `46583684…` (summary §5.1) |
| DreaMS | `pluskal-lab/DreaMS` @ `dbec3a0` (local `DreamS/`) | novel worker, **GPU 1** | Upstream pins numpy 1.25.0 (no Python 3.12 wheel), pytorch-lightning 2.0.8, torch 2.2.1 and a git dependency `msml` (`DreamS/setup.py:27-52`). **Decided (A2): vendored.** Kaggle runs `build/dreams_infer.py` (torch + numpy only) on the converted `dreams_embed.pt`. The full package is installed **on Colab only**, for the one-off conversion in P0.1a. |
| MassSpecGym | `pluskal-lab/MassSpecGym` @ `f259fe3` (local `MSGYM_novel/`) | **Colab training only** | Pins torch 2.3.0, numpy 1.24.4 and Python ≥ 3.11 (`MSGYM_novel/setup.py`), so it is installed with `--no-deps`. (A5) The import chain then needs `pytorch-lightning`, `torchmetrics`, `torch_geometric`, `myopic-mces`, `pulp` (with a solver), `matchms`, `chemparse`, `tokenizers`, `selfies`, and `distutils` via setuptools. **Not installed on Kaggle**: the vocabulary ships as JSON, and the model class lives in `build/novel_decoder.py`. |
| selfies | pure-Python wheel | novel worker + Colab | version **to pin in P0**; the same version on both sides (grammar parity) |
| Novel wheelhouse | dataset `casmi26-novel-offline` | Kaggle input | holds `dreams_embed.pt`, `parity_200.npz`, the `selfies` wheel and `manifest.json`. The vendored DreaMS code is embedded in the notebook, not shipped. Built in P0.1a. |

**Hardware.** Kaggle gives 2× T4, 4 vCPU, 9 h and no internet (summary §1). ICEBERG measured
36.6 mol/s on one T4. A second T4 added nothing to ICEBERG, because it is CPU-bound (summary §5.3).
GPU 1 therefore goes to the novel worker. Whether its CPU load starves ICEBERG is **to measure** (P5).

**One model definition.** `build/novel_decoder.py` holds the decoder class, the vocabulary mapper,
the grammar tracker and the search. The Colab training notebooks and the Kaggle worker both import
it (plan §5.5). The class subclasses MassSpecGym's `DeNovoMassSpecGymModel` only on Colab, through a
thin wrapper, so that the Kaggle import needs no massspecgym.

---

## 2. Artefacts

| artefact | content | produced by | size |
|---|---|---|---|
| `embedding_model.ckpt` | DreaMS `ContrastiveHead` (1024-d precursor-token embedding) | Drive `casmi_2026/DreamS_Embedding_Model`; live source HF `roman-bushuiev/DreaMS`, archive Zenodo 10997887 (A1) | SHA1 **to record** |
| `ssl_model.ckpt` | DreaMS backbone. It is the only place the architecture `args` are stored (A2). | HF `roman-bushuiev/DreaMS` | **Colab only**, for the P0.1a conversion; SHA1 recorded |
| `dreams_embed.pt` | plain `state_dict` (backbone + head, training heads dropped), backbone `args` dict, DreaMS commit, source SHA1s | P0.1a conversion | SHA1 recorded and asserted at load |
| `parity_200.npz` | 200 stratified corpus peak lists, their row ids, and the Colab vendored embeddings | P0.1a | the G0.1 reference and a P5 test fixture |
| `_dreams_out/emb/emb_*.npy` + `corpus_meta.parquet` | cached 1024-d float32 embeddings for the filtered train corpus | `casmi_dreams_colab.ipynb` Stage 3 | on Drive; rows **to report in P0.2** |
| `corpus_ce.parquet` | CE per cached row, recovered by replaying Stage 3a | P0.2 | written; 1,323,878 rows, CE missing on 23.9 % |
| `selfies_vocab.json` | SELFIES symbol → token ID, plus the special tokens `<pad> <s> </s> <unk>`. The alphabet is the corpus alphabet ∪ the robust alphabet, with stereo stripped (A4, A9). | P0.3 | SHA1 recorded |
| `msg_ik14.npy` | sorted ik14s of every `MassSpecGym1.5.tsv` SMILES | P0.5 | the ICEBERG `msg_all` exclusion set |
| `_novel/emb_dbec3a0_lx/emb_*.npy` + `manifest.json` | corpus embeddings rebuilt with the fixed (lexsort) `dreams_peaks`, `dreams_infer` SHA1 `57ce1bc1…`; row `i` = `corpus_meta` row `i` | G0.1-R2 (`casmi_novel_lx_reembed_colab.ipynb`) | **the P3 training cache** once G0.1-R2 passes; `emb_dbec3a0` is provisional |
| `parity_lx.npz` | 200 parity rows (150 with an intensity tie at the top-100 cut): row ids, `train_row`, peaks, identities, new-cache vectors | G0.1-R2 | the cross-platform reference; replaces `parity_200.npz` as the P5 fixture |
| `pseudo_c3.parquet`, `c2_displace.parquet` | one row per molecule: `molecule_id`, `inchikey14`, `smiles`, `can`, `selfies`, `formula`, `scaffold`, `group_size`, `src_lib`, `timstof`, `in_scope`, `rows` (corpus rows = embedding rows), `train_rows`, `adducts`, `ces`, `lib_ban_keys`, `pool_ban_keys`, `dreams_ft_risk` | P1 (`casmi_novel_p1_colab.ipynb`) | **to measure** (G1) |
| `dec_split.parquet` | `inchikey14`, `scaffold_key`, `split` ∈ {`train`, `dec_val`, `heldout_c3`, `heldout_c2`} | P1 | — |
| `decoder.pt` | `state_dict`, hyperparameters, vocab SHA1, adduct list, CE bins | P3 | **to measure** |
| `gate_report.md` | pre-registered thresholds and measured results per gate | P0–P6 | — |

Existing inputs stay unchanged: the competition data, COCONUT (`coco_meta.pkl`, `coco_mass.npy`,
`coco_fp.npy`), `fp_bits.npy` and `casmi26-iceberg-offline` (summary §5.4).

---

## 3. Reuse

| from | used for |
|---|---|
| `simsim_common.neutral_mass1`, `precursor_of`, `ADDUCT_DELTA`, `TEST_ADDUCTS` | neutral mass per adduct; the theoretical precursor of novel candidates |
| `simsim_common.formula_mass`, `PPM_WIN`, `PPM_CENTER`, `TOL` | formula enumeration window (8.5 ppm), centring (1.45 ppm), fragment tolerance |
| `simsim_common.prep`, `esim`, `spectrum_sv` | `sv` of novel candidates, computed exactly as for pool candidates |
| `simsim_common.build_requests`, `run_worker`, `read_response`, `sha1` | ICEBERG requests for novel candidates; plumbing for the novel worker |
| `simsim_common.rank_two_tier` | the base order that the slot policy merges into |
| `build/iceberg_worker.py` | template for `build/novel_worker.py` (JSON in and out, device argument, batched) |
| `build/py2nb.py`; the `build/test_simple_sim.py` fake-worker pattern | notebook generation; `test_novel_sim.py` |
| `casmi_dreams_colab.ipynb` `_dreams_peaks` and Stage 7 packaging | the DreaMS preprocessing (moved into `build/` as a single shared function) and the offline dataset layout |
| MassSpecGym `SelfiesTokenizer` (`models/tokenizers.py:58`) | building the vocabulary (Colab); exported to JSON |
| MassSpecGym `MolToFormulaVector` (`data/transforms.py:270`) | formula count vector for the formula token (element order = `CHEM_ELEMS`) |
| MassSpecGym `DeNovoMassSpecGymModel` metrics (`models/de_novo/base.py`) | top-k exact match, MCES and Tanimoto in P3/P4 |
| MassSpecGym `RandomDeNovo` (`models/de_novo/random.py:180`) | null baseline for G3 |

New files:

- **P5:** `build/novel_decoder.py`, `build/novel_worker.py`, `build/novel_sim_submit.py`,
  `build/mk_novel_sim.py`, `build/test_novel_sim.py`.
- **P0–P4 Colab notebooks:** generated in the same way (`build/mk_novel_p*.py`).

---

## 4. Interfaces

```python
# --- formula stage (main kernel) ---
def enumerate_formulas(neutral_mass: float, ppm: float = PPM_WIN,
                       elements=("C","H","N","O","P","S","F","Cl","Br","I")) -> list[str]
def score_formulas(formulas: list[str], spectra, window_formula_counts) -> list[tuple[str, float]]

# --- novel worker (GPU 1) ---
def dreams_peaks(mz, inten, prec) -> np.ndarray | None        # (2, ≤100); the shared preprocessing
def dreams_embed(peak_lists, precursors) -> np.ndarray         # (n, 1024)
def build_memory(embs, adducts, ces, formula) -> torch.Tensor  # (k+1, D_MODEL), plan §3.5

class Generator(Protocol):
    def search(self, memory, formula: str, budget: int) -> list[tuple[str, float]]
    # (smiles, logp); SelfiesDfsGenerator now; analog decoration later (P7)

class Tracker:                                  # plan §4.1–4.2
    def __init__(self, table: TokenTable, formula: dict | None): ...   # {element: count} incl. H
    def legal_mask(self) -> np.ndarray          # bool over the vocabulary (torch-free; the search converts)
    def push(self, token_id: int) -> "Tracker"  # returns the new state (persistent, cheap copy)
    def eos_legal(self) -> bool

# --- main kernel ---
def check(smiles: str, formula: str) -> str | None
    # ik14 if: sanitizes, single component, FULL formula incl. H == target; else None
def slot_merge(base: list[str], novel: list[tuple[str, float, float]], policy: str,
               k_novel: int, delta: float) -> list[str]
    # ik14-deduped, len <= 25; "tail" (v1) or "arena" (v2), plan §3.8
```

### 4.1 Per-molecule record (extends Approach B's)

```python
novel = {
  "routed": bool,                       # False: max lv >= τ_gen (or adduct not in GEN_ADDUCTS)
  "formulas": [(formula, score)],
  "funnel": {"leaves": int, "valid": int, "single": int, "formula_ok": int,
             "in_pool": int, "novel_unique": int},
  "candidates": [{"smiles": str, "ik14": str, "formula": str,
                  "logp": float, "freq": int, "sv": float, "fr": float}],
  "embed_seconds": float, "search_seconds": float,
}
```

`run_report.json` aggregates the funnel and the routing counts, so a fast commit shows whether
generation ran at all.

---

## 5. Parameters

Every value starts as provisional and is set by the named phase. None is tuned on the visible test,
which consists of training-set examples (data page).

| parameter | provisional | set by | meaning |
|---|---|---|---|
| `USE_NOVEL` | True / False | — | the only difference between treatment and control |
| `τ_gen` | 0.75 | P6 probe | skip generation if max `lv` ≥ τ_gen |
| `GEN_ADDUCTS` | all 10 test adducts | P4 | restricted only if P4 shows an adduct is pure cost |
| `N_PEAKS_DREAMS` | 100 | fixed | the checkpoint's `n_highest_peaks` (`api.py:108`) |
| `MAX_LEN` | **171** (p99.9 of corpus SELFIES lengths, P0.3) | P0 | search depth limit |
| `MAX_SPEC_PER_MOL` | **to measure** | P3 | spectra per molecule, sampled per epoch in training; all spectra (up to 16) at test |
| `D_MODEL`, `N_DEC_LAYERS`, `N_HEADS` | **to measure** | P3 | decoder size, bounded by Colab time |
| `CE_BINS` | **[15, 23, 36, 44]** (P0.2 quantiles; 23.9 % of rows go to "unknown") | P0 | CE histogram bins, plus "unknown" |
| `FORMULA_TOPK` | 3 | P2 / P4 | formulas searched per molecule |
| `BRANCH` | **to measure** | P4 | tokens expanded per node |
| `NODE_BUDGET` | **to measure** | P4 | expansions per (molecule, formula) |
| `MAX_LEAVES` | 50 | P4 | leaves emitted per (molecule, formula) |
| `DBE_PRUNE` | **False**: P0.4 found 139 DBE violations (A8) | P0.4 | plan §4.2 soundness rule |
| `MAX_NOVEL` | 50 | P4 | novel candidates sent to ICEBERG per molecule |
| `SLOT_POLICY` | `"tail"` | P6 | `"tail"` (v1) or `"arena"` (v2) |
| `K_NOVEL` | 5 | P4 / P6 | tail slots for novel candidates |
| `DELTA` | **from P4** | P4 / P6 | arena margin in `sv` |
| `GEN_BUDGET_H` | 3.0 | P4 / P5 | wall-clock cap on generation |

---

## 6. Runtime budget

```
T_total = T_base + T_novel_install
        + N_routed · ( T_embed + FORMULA_TOPK · T_search(NODE_BUDGET) )
        + N_routed · MAX_NOVEL / r_iceberg
```

| term | value | source |
|---|---|---|
| `T_base` (Approach B full run) | finished inside 9 h; the exact hidden-run time is unknown | summary §5.6 |
| ICEBERG offline install | 197 s | summary §5.3 |
| `r_iceberg` | 36.6 mol/s, one T4, batch 128 | summary §5.3 |
| ICEBERG worker start-up | 10.4 s per call | summary §5.3 |
| `T_novel_install` (DreaMS + selfies) | 3.7 s (selfies wheel; DreaMS is vendored, nothing to install) | P0.1b |
| `T_embed` per spectrum (DreaMS, T4) | 6.68 ms at batch 64 on `cuda:1`; 5.0 s to load `dreams_embed.pt` | P0.1b |
| `T_search(NODE_BUDGET)` per formula (T4, with KV cache) | **to measure** | P4 |
| `N_routed` | **to measure** on the visible test; the hidden split is unknown | P5 |
| extra ICEBERG time | at `MAX_NOVEL` = 50 and 400 molecules: ≤ 20,000 requests ≈ 9.1 min at 36.6 mol/s. This is arithmetic from the measured rate and excludes the CE count per spectrum. | derived |

The novel worker runs on GPU 1 while ICEBERG is on GPU 0. Their overlap on 4 vCPU is **to measure**.

**Watchdog order.**
1. Generation stops at `GEN_BUDGET_H`.
2. ICEBERG keeps its existing budget and stop time (7.5 h).
3. Unfinished molecules keep their base lists.

---

## 7. Licences

| item | licence | status |
|---|---|---|
| MassSpecGym code | MIT | verified: `MSGYM_novel/LICENCE` |
| MassSpecGym data (HF `roman-bushuiev/MassSpecGym`, incl. the 4M molecule set) | MIT | memory: simulator-checkpoint-licence |
| DreaMS code + weights | MIT; weights on HF `roman-bushuiev/DreaMS` and Zenodo 10997887 (public, no licence gate) | `DreamS/setup.py:23`, README badges. Fine-tuning data is provisionally MoNA `[M+H]+` (`training/fine_tune.sh:45-71`); P0.1a confirms it. |
| ICEBERG code + `msg_all` | MIT; MassSpecGym-trained | summary §5.1, G1 pass |
| selfies | Apache-2.0 | **to confirm in P0** against the pinned version |
| RDKit | BSD-3-Clause | — |
| COCONUT 2.0 | CC0 / CC-BY | as in the existing specs |

NIST-trained weights are not allowed, because they are not freely available. The DreaMS weights are
public; if the head was fine-tuned on NIST20 pairs, that is recorded as a leakage caveat and not as a
licence problem.
