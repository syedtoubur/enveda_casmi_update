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

## Kaggle datasets for R1 (pool build inputs)
`casmi26-v2-pool` (710,701 structures), `chebi-lipidmaps-casmi26` (`bio_*`), `casmi26-natural-product-knowledge-table`
(`natural_products.parquet`, 152,717 rows; 2,715 not in the pool), `casmi26-pubchem-tier` (105.9M rows) +
`casmi26-pubchem-popularity-prior` (`pc_np` bits: 1 COCONUT, 2 LOTUS, 4 NPAtlas). See `specs/Dataset_intro/`.
