# Data dictionary

All JSONL files contain one UTF-8 JSON object per line. Missing raw source strings stay JSON `null`; they are not guessed or normalized. Field values come from the pinned public reference compilation or the original declared numerical transformation. No competition membership field is present.

## `record_attribution.jsonl`

| Field | Meaning |
|---|---|
| `source_row_id` | Zero-based row position in the pinned 139,744-row source `spectra.parquet`; not a molecular identifier |
| `spectrum_id` | Supplied reference spectrum ID; preserved verbatim |
| `inchikey` | Full supplied key; no case change, truncation or tautomer canonicalization |
| `full_inchikey_valid` | Syntax check for the full 27-character key form; chemical validity is unmeasured |
| `ion_mode`, `adduct`, `collision_energy` | Raw source strings; adduct semantics, collision units and missing values remain uninterpreted |
| `source`, `source_institution` | Raw source attribution |
| `license` | Raw source licence string; all admitted rows equal exactly `CC0` |
| `authors`, `publication`, `copyright` | Original raw attribution; preserve these when reusing a record |
| `supplied_spectrum_id_present` | Whether the supplied ID is a nonempty string |
| `supplied_spectrum_id_seen_before` | Whether that supplied ID occurred earlier among admitted source occurrences |

The derivative's compilation terms remain NC-SA4 despite the per-record CC0 label.

## `reference_molecules.jsonl`

| Field | Meaning |
|---|---|
| `inchikey`, `full_inchikey_valid` | Full exact source key and syntax check |
| `source_structure_row_id` | Zero-based row position in the pinned 20,204-row source structure table; not an ID across releases |
| `supplied_exact_mass` | Positive finite source value when qualified numerically; it is not a verified neutral-mass estimate |
| `supplied_exact_mass_source_repr` | Original parsed float/null representation for audit, including invalid values if any |
| `supplied_mass_issue` | JSON `null` for a finite positive value; text `"null"` for a missing source value, `"non_finite"` or `"non_positive"` for those issues |
| `reference_neutral_mass_basis_verified` | `false` throughout this release |
| `source_key_row_multiplicity` | Number of source structure rows sharing the exact full key; every row is retained |
| `source_key_mass_conflict` | Whether source rows for that key carry different supplied mass representations |
| `admitted_spectra_occurrences` | Number of literal-CC0 source occurrences for the full key |
| `distinct_admitted_spectrum_ids` | Count of distinct nonempty supplied IDs; missing values are not IDs |
| `distinct_nonempty_admitted_institutions` | Distinct nonempty raw institution labels |
| `distinct_admitted_raw_conditions` | Distinct raw `(key, ion_mode, adduct, collision_energy, source_institution)` tuples; missing fields remain part of the raw tuple |
| `coconut_row_count_at_supplied_mass_3ppm`, `..._10ppm`, `..._20ppm` | Count of COCONUT source rows in the closed interval around the supplied mass |

For source value `q` and declared tolerance `p`, the numerical interval is:

```text
[q - q*p/1,000,000, q + q*p/1,000,000]
```

The lower search uses `side='left'`, the upper search `side='right'`. Repeated candidate masses remain separate source rows. Counts apply no cap or ranking. They do not prove a structure is present, molecular uniqueness or query coverage. A chemically interpreted neutral window needs qualified producer mass/charge/salt semantics.

## `condition_multiplicity.jsonl`

The fields `inchikey`, `ion_mode`, `adduct`, `collision_energy` and `source_institution` define an exact raw tuple. `admitted_source_occurrences` is its occurrence count. Counts sum to 6,951. No collision-energy unit conversion, adduct mapping or polarity inference was performed.

## `UNMATCHED_REPORT.json`

The actual unmatched count is zero. The report records that fact and the byte/hash identity of the original empty JSONL diagnostic. It contains no synthetic reference row. A future build with unmatched keys must preserve its actual diagnostic rows.

## Manifests and checksums

`MANIFEST.json` preserves the original build's exact source pins, scope flags and runtime versions. `PUBLIC_VALIDATION.json` describes executed checks and measured resource use. `RELEASE_FILES.json` pins downloadable files; it excludes itself and `dataset-metadata.json`, which is native publication configuration consumed by the creator.
