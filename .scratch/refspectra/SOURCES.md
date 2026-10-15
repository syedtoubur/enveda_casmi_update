# Sources and reproducibility

## Reference compilation

[Samar Talwar: CASMI 2026 Spectral Library (MassBank, harmonized)](https://www.kaggle.com/datasets/samartalwar/casmi-2026-spectral-library-massbankharmonized), numeric dataset ID **12067949**, public version **1**. The current primary card was inspected on 2026-10-12 and identifies CC BY-NC-SA 4.0 compilation terms, mixed per-record licences, preserved attribution and publisher-excluded ND records.

The source derives from [MassBank/MassBank-data](https://github.com/MassBank/MassBank-data). Credit its contributing institutions and each record's preserved authors/publication/copyright. The atlas's extraction uses only the exact raw `license == "CC0"` subset and retains compilation NC-SA scope.

| Exact public input | Bytes | SHA-256 |
|---|---:|---|
| `spectra.parquet` | 43,941,912 | `6ebb46b32ddd344d988c3bdb90e7160d4ce46e6d0b900a1217607aa2234c9477` |
| `structures_with_fingerprints.parquet` | 3,043,054 | `8b7a0014184e436e19b13cb526b8fafebd5c967c8f36fb2c0eed716e462c7279` |

The full source has 139,744 reference rows and 20,204 structure rows. Only eleven metadata columns from the reference file and full key plus supplied `exact_mass` from the structure file were projected. Peak arrays, SMILES, fingerprints and all membership flags were excluded. The presence of other columns in a source file is not evidence that their values were read.

## Candidate mass source

[prvsiyan: COCONUT candidates](https://www.kaggle.com/datasets/prvsiyan/coconut-casmi26-candidates), numeric dataset ID **12039473**, preserved public version **2**. Credit the [COCONUT team](https://coconut.naturalproducts.net/). Retain the admitted snapshot's NC-SA4 derivative scope.

| Exact public input | Bytes | SHA-256 |
|---|---:|---|
| `coco_mass.npy` | 3,491,240 | `2b2af92c3185fa8cd52a4a3e3e7fd7eda4d94e8b9402662a3a4c55ffb13c6e5c` |

This is the previously verified finite-positive, nondecreasing 436,389-row float64 mass array. The new build checks that contract again before search. No companion pickle, selected fingerprint indices, fingerprint bank or model weights were used.

## Original transformation

1. Verify exact input byte pins and source counts/schema.
2. Admit only literal-CC0 reference occurrences, retaining raw labels and complete attribution.
3. Validate full 27-character keys. Preserve malformed/unmatched diagnostics; only valid exact keys join.
4. Preserve all matching source structure rows and detect contradictory or invalid supplied masses.
5. Count every COCONUT mass row in closed ±3/10/20 ppm intervals around each valid supplied source mass. No cap or ranking is applied.
6. Group admitted occurrences by raw condition tuples. Preserve raw source row positions and repeated-ID diagnostics.

The exact reviewed build source SHA-256 is `77f47345b3495b07f526c75b063cf0bbf8d655224db959e77a9cd62b98872574`; eight authored boundary controls used SHA-256 `ab835d103fae35bef526e2cb6c4b32afcba3a3f7ffdd9c6b5f0a8f09a7c7397f`. The controlled builder and private environment evidence are preserved locally. The public measured validation binds its source and actual receipt hashes; no private runtime inventory is distributed or required to consume these tables.

The standard-library readers in this dataset support independent local inspection. `rebuild_reference_evidence.py` is a separate original portable variant that removes the local private runtime-inventory prerequisite. It retains the reviewed public source pins, projected columns, numerical/data transformations and conservative limits. Its source was reviewed separately; no second real derivation was performed. The public helper uses installed NumPy 2.2.6 and PyArrow 25.0.0 and accepts explicit input paths and a new output directory. Its generated receipt belongs to its own execution, not the older local build.

For a new derivation, record your own input hashes, row-admission rule, implementation, dependency versions and actual validation; do not adopt an old receipt as proof of a changed build. The manifest's Python version can differ on another interpreter, so the entire output directory is not promised to be byte-identical without a new measured comparison.

Original environment: Python 3.12.14, NumPy 2.2.6, PyArrow 25.0.0. The actual build's measured resource use and direct-scan controls are in `PUBLIC_VALIDATION.json`. No model, competition runtime or leaderboard evaluation was executed.

## Related resources

- [Original candidate interoperability kit](https://www.kaggle.com/datasets/prvsiyan/casmi-candidate-interop-kit)
- [Original COCONUT Proton Ion Atlas](https://www.kaggle.com/datasets/prvsiyan/coconut-proton-ion-mass-index)
- [Aryan Awasthi's COCONUT Morgan/MACCS candidates](https://www.kaggle.com/datasets/aryanawasthiteykik/casmi26-coconut-morgan-maccs-fingerprints)
- [Aryan Awasthi's spectrum-to-fingerprint MLP](https://www.kaggle.com/datasets/aryanawasthiteykik/casmi26-spectrum-to-fingerprint-mlp)

Aryan's candidate/model resources motivated explicit compatibility documentation; their bytes were not used by this build. Their publisher credits prvsiyan's candidate bank and COCONUT. Keep that source lineage when combining related work.
