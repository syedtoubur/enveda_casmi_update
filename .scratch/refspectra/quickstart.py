"""Inspect downloaded atlas data locally or from a Kaggle input directory."""
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 cubres
import argparse
import json
from reference_evidence import evidence_for_key, read_reference_records, verify_downloadable_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()
    pins = verify_downloadable_files(args.data_dir)
    rows = list(read_reference_records(args.data_dir))
    if not rows:
        raise ValueError("atlas contains no qualified source structure rows")
    first = rows[0]
    evidence = list(evidence_for_key(args.data_dir, first["inchikey"]))
    if len(evidence) != first["admitted_spectra_occurrences"]:
        raise ValueError("reference evidence multiplicity differs")
    print(json.dumps({"file_pins": pins, "source_structure_rows": len(rows),
                      "example_source_key": first["inchikey"],
                      "example_supplied_mass": first["supplied_exact_mass"],
                      "example_numeric_rows_at_10ppm": first["coconut_row_count_at_supplied_mass_10ppm"],
                      "example_reference_occurrences": len(evidence),
                      "neutral_mass_basis_verified": first["reference_neutral_mass_basis_verified"]}, indent=2))


if __name__ == "__main__":
    main()
