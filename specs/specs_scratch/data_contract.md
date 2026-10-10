# Data contract

## Competition files (`enveda-CASMI26-molecule-id-mass-spectra/`)
| File | Size | Notes |
|---|---|---|
| `train.parquet` | 3.03 GB, 2,539,608 spectra, 275,810 ik14 | Libraries: enveda-180 1.15M (synthetic drug-like, timsTOF), pluskal_ms2 528k, riken 347k, gnps 221k, massbank 102k, mona 92k, spectraverse 51k, msdial 41k, drug_plus 2.5k, enveda-np-examples 1.2k (250 NP mols, timsTOF), masaryk 0.7k. |
| `test.parquet` | 1,213 spectra / 400 molecules (public) | The **public** file is 1,213 verbatim enveda-180 rows — synthetic, not NP. The hidden rerun file is ~1,500 spectra / ~400 NP molecules. The visible commit score is meaningless. |
| `sample_submission.csv` | | `molecule_id, smiles` (25 `;`-separated SMILES). |

### test columns (12)
`molecule_id, spectrum_id, ms2_mzs, ms2_normalized_intensities, base_peak_intensity, adduct, ionization_mode,
instrument_type, precursor_mz, collision_energy_orig, collision_energy_ev, collision_energy_orig_units`

### train columns
The test columns minus `molecule_id, spectrum_id`, plus `ingest_lib, normalized_smiles, inchikey, inchikey14,
molecular_formula, adduct_orig, precursor_error_ppm, num_peaks`. Nulls: base_peak_intensity 45.8%,
collision_energy_orig 12.6%, instrument_type 1.3%.

### Public test profile
Adducts: [M+H]+ 959, [M-H]- 193, [M+CH2O2-H]- 31, [M+Na]+ 22, [M+NH4]+ 4, [M+K]+ 2, [M+Cl]- 2.
CE: 60 (363), 40 (362), [20,40,60] (298), 20 (190); CE known 100%; merged 24.6%. Spectra/molecule median 3.
All timsTOF, precursor reads +1.45 ppm high (`PPM_CENTER`).

## Kaggle datasets used by R0
| Dataset | Files used | Owner |
|---|---|---|
| competition data | train / test / sample_submission | Enveda |
| `casmi26-iceberg-offline` | `wheelhouse/*.whl`, `manifest.json`, `ckpt/gen.ckpt`, `ckpt/inten.ckpt` (MassSpecGym `msg_all`) | ours (packaged third-party weights) |
| COCONUT arrays (phase1) | `coco_meta.pkl` (`keys`, `smiles`, `nbits`), `coco_mass.npy` | third-party |
| `casmi26-offline-rdkit-2026033` | `rdkit-*.whl` | ours |

R0 no longer needs `fp_bits.npy` / `coco_fp.npy`: the base computed fingerprints it never used.

**2026-10-06: Kaggle image change.** The `cp311` RDKit wheel is rejected on the current image. All Kaggle twins now
install like the v18 zenith notebook:
- **Main interpreter RDKit:** the attached `rdkit-*.whl` built for the image's Python tag
  (`eveliaveldrine/rdkit-wheels`). A matching 2026.03.3 build is preferred if one exists, otherwise 2025.3.6. R1 pool
  keys are precomputed with 2026.03.3, so this affects only MetFrag-lite. `run_report.json` records the version.
- **ICEBERG:** if `casmi26-iceberg` is attached (`ice_runner.py`), `sim.setup_ice_pkg` installs its `wheels/*.whl.ice`
  minus its RDKit, plus the matching-tag 2025.3.6 build, into `--target ice_site`. Our `iceberg_worker.py` then runs
  under the notebook's Python with `PYTHONPATH = shim/ : src/ : ice_site`, using the `ckpt/` files whose names
  contain gen / inten. Otherwise it falls back to the `casmi26-iceberg-offline` venv. If both are attached, the
  offline manifest checks the checkpoint SHA1s.

## Kaggle datasets for R1 (pool build inputs, read from Drive `casmi_2026/` on Colab)
`casmi26-v2-pool` (`pool_meta.parquet`, 710,701 structures), the COCONUT arrays R0 uses (`coco_meta.pkl`),
`chebi-lipidmaps-casmi26` (`bio_meta.pkl`), `casmi26-natural-product-knowledge-table` (`natural_products.parquet`,
152,717 rows; 2,715 not in the pool), and `train.parquet`. No PubChem tier and no popularity arrays in R1.
See `specs/Dataset_intro/`.

## `casmi26-scratch-pool` (ours; built by `casmi_scratch_r1_pool_colab.ipynb`)
| File | Content |
|---|---|
| `pool_mass.npy` | float64 neutral exact mass (RDKit `ExactMolWt` of the kept SMILES), ascending |
| `pool_key.npy` | S14 scorer_key (RDKit 2026.03.3 canonical-tautomer InChIKey14), unique |
| `pool_flags.npy` | uint16 provenance bits: 1 train, 2 enveda-180, 4 other train library, 8 COCONUT (R0 arrays), 16 v2-pool, 32 BIO, 64 NP table, 128 LOTUS, 256 NPAtlas. enveda-180-only = bit 2 and none of 4/8/32/64 (`pool.e180_only`) |
| `pool_smiles.parquet` | `smiles`, verbatim from the winning source (train > v2 > COCONUT > BIO > NP table) |
| `pool_np.parquet` | `key`, `np_pathway`, `np_superclass`, `np_class`, `label_source` (0 none, 1 table, 2 kNN ≥ 0.7), `knn_sim`, `kingdoms`, `n_organisms`, `np_score` (Ertl) |
| `train_keymap.parquet` | train `inchikey14` → `scorer_key`; R1 runs remap the library with it so `lv` looks up pool keys |
| `MANIFEST.json` | version, RDKit, inputs, per-source counts, unkeyable drops, key disagreements, R1.1, label stats, sha1 per file |
| `pool_npc.parquet` (**v2**, `casmi_scratch_r2_npc_colab.ipynb`) | row-aligned with `pool_key.npy`: `key`, NPClassifier top-3 superclasses `sc1..sc3` + probabilities `sp1..sp3`, top-1 `pathway` / `pathway_p`, `class` / `class_p` (null where RDKit cannot parse). MANIFEST v2 adds an `npc` block (spike, counts, agreement with table labels) |

Kaggle twins `casmi_scratch_r1a.ipynb` / `r1b.ipynb` attach it **instead of** the COCONUT arrays. The R2 twins
(`casmi_scratch_r2_ice` / `r2_gl`, `class_vote=True`) need **v2** and fail loudly without `pool_npc.parquet`.

## `casmi26-glacier` (Ahmed Berat Özer; `/kaggle/input/datasets/ahmedberatozer/casmi26-glacier`)
GLACIER isomer simulator: `gl_runner.py`, `gl_fuse.py`, `ckpt/glacier.pt` (MassSpecGym weights, MIT; sha256 in
`MANIFEST.json`, checked by `sim.setup_glacier`), `shim/` (dgl / torch_scatter / LinSATNet stand-ins), `src/ms_pred/`.
It ships **no wheels**: `casmi26-iceberg` (its `wheels/*.whl.ice`) and `eveliaveldrine/rdkit-wheels` (RDKit 2025.3.6
for the image's cp tag) must stay attached to build the RDKit 2025.03 site. Only the GLACIER twins
(`casmi_scratch_r1a_glacier` / `r1b_glacier`, `sim_model="glacier"`) attach it. Our `resources/glacier_worker.py` imports
`gl_runner.py` as a library and speaks the ICEBERG worker's TSV -> NPZ protocol.


## R3 FPNet inputs (attached as-is, not built by us)
| Dataset | Files used | Notes |
|---|---|---|
| `casmi-fpnet` (ours) | `fp_single_s0.pt`, `fp_single_s1.pt`, `fp_merged_m0.pt`, `fp_merged_m1.pt`, `fp_bits_sha1.txt` | family P; found by `fp_bits_sha1.txt`, not by dataset name; has **no** `fp_bits.npy`, so the COCONUT arrays dataset must be attached too |
| COCONUT arrays dataset | `fp_bits.npy` (6,930), `coco_fp.npy`, `coco_meta.pkl` | own `fp_bits.npy` is picked by sha1, never by path; `coco_*` only for the parity report |
| `casmi26-fpnet-full1` (Ahmed Berat Özer) | `fpnet_full1.pt` | family A; CC BY-NC (derived from competition data) |
| `casmi26-v2-pool` (Ahmed Berat Özer) | `fp_bits.npy` (10,226), `pool_meta.parquet` (`smiles`), `pool_fp.npy` | `fp_bits.npy` picked by length = checkpoint nbits; parity asserted (≥ 45/50) |

Candidate fingerprints are computed at runtime from `pool_smiles` (scored range + stage B only). The family-A model code
(`FPNet`, `prep_peaks`, `ADDUCT_LIST`, `raw_fingerprint`, `merge_spectra`) is vendored in `casmi_scratch/fpnet.py`
from the `casmi26-v4b-models` code (`casmi/fpnet.py`, `casmi/chem.py`, `casmi/spectra.py`).

## R4 FPNet input (added to R3's)
| Dataset | Files used | Notes |
|---|---|---|
| `prvsiyan/casmi26-fp-models-v2` | `fp_single_s2.pt`, `fp_merged_m1.pt` | family P (`kind="prv"`); found by the dataset path; no `fp_bits_sha1.txt`, so the COCONUT arrays dataset (`prvsiyan/coconut-casmi26-candidates`: `fp_bits.npy` 6,930 next to `coco_fp.npy`) must be attached; parity asserted (≥ 45/50) against `coco_fp.npy` |

