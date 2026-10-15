"""Original portable rebuild of the reference evidence atlas.

This separate variant preserves the reviewed numerical/data projection core.
It accepts only the three public source byte pins and a new output directory.
Use Python 3.10+ on Linux/macOS, NumPy 2.2.6 and PyArrow 25.0.0.
No private environment inventory, SDK, network or credential is required.
The published local-build receipt does not certify execution of this variant.
"""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 cubres

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import resource
import signal
import sys
import time

PINS = {
    "spectra": (43941912, "6ebb46b32ddd344d988c3bdb90e7160d4ce46e6d0b900a1217607aa2234c9477"),
    "structures": (3043054, "8b7a0014184e436e19b13cb526b8fafebd5c967c8f36fb2c0eed716e462c7279"),
    "mass": (3491240, "2b2af92c3185fa8cd52a4a3e3e7fd7eda4d94e8b9402662a3a4c55ffb13c6e5c"),
}
SPECTRA_COLUMNS = (
    "spectrum_id", "inchikey", "ion_mode", "adduct", "collision_energy",
    "source_institution", "source", "license", "authors", "publication", "copyright",
)
STRUCTURE_COLUMNS = ("inchikey", "exact_mass")
TOLERANCES = (3.0, 10.0, 20.0)
INCHIKEY = re.compile(r"[A-Z]{14}-[A-Z]{10}-[A-Z]\Z")
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
MAX_RSS_BYTES = 256 * 1024 * 1024
WALL_SECONDS = 90
CPU_SECONDS = 60
PUBLIC_RUNTIME_VERSIONS = {"numpy": "2.2.6", "pyarrow": "25.0.0"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def key_valid(key):
    return isinstance(key, str) and INCHIKEY.fullmatch(key) is not None


def mass_bounds(center, ppm):
    if not math.isfinite(center) or center <= 0:
        raise ValueError("supplied mass must be finite and positive")
    if not math.isfinite(ppm) or ppm < 0:
        raise ValueError("ppm must be finite and nonnegative")
    width = center * (ppm / 1_000_000.0)
    lower, upper = center - width, center + width
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError("window endpoints must remain finite")
    return lower, upper


class Budget:
    """CPU/wall signals, batch-boundary RSS checks, exact output-byte cap.

    RSS is measured at checkpoints; it is not a claimed OS-enforced memory limit.
    Use an external memory supervisor if a hard RSS cap is required.
    No input/output cleanup is performed on failure.
    """

    def __init__(self):
        self.started = time.monotonic()
        self.cpu_started = time.process_time()
        self.bytes_written = 0
        signal.signal(signal.SIGALRM, self.stop)
        signal.signal(signal.SIGXCPU, self.stop)
        signal.setitimer(signal.ITIMER_REAL, WALL_SECONDS)
        previous = resource.getrlimit(resource.RLIMIT_CPU)
        requested = math.ceil(self.cpu_started + CPU_SECONDS)
        hard = previous[1]
        soft = requested if hard == resource.RLIM_INFINITY else min(requested, hard)
        resource.setrlimit(resource.RLIMIT_CPU, (soft, hard))

    @staticmethod
    def stop(signum, frame):
        raise RuntimeError("derivation time budget exceeded")

    def rss(self):
        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(raw if sys.platform == "darwin" else raw * 1024)

    def check(self):
        if time.monotonic() - self.started > WALL_SECONDS:
            raise RuntimeError("wall budget exceeded")
        if time.process_time() - self.cpu_started > CPU_SECONDS:
            raise RuntimeError("CPU budget exceeded")
        if self.rss() > MAX_RSS_BYTES:
            raise RuntimeError("RSS budget exceeded at checkpoint")
        if self.bytes_written > MAX_PAYLOAD_BYTES:
            raise RuntimeError("payload budget exceeded")

    def write_json(self, stream, value, *, line=False):
        text = json.dumps(value, ensure_ascii=False, allow_nan=False,
                          sort_keys=True, indent=None if line else 2) + "\n"
        data = text.encode("utf-8")
        if self.bytes_written + len(data) > MAX_PAYLOAD_BYTES:
            raise RuntimeError("payload budget would be exceeded")
        stream.write(data)
        self.bytes_written += len(data)
        self.check()


def verify_runtime():
    """Portable installed-library versions; no local path/inventory dependency."""
    if sys.version_info < (3, 10):
        raise ValueError("Python 3.10+ is required")
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    if (np.__version__ != PUBLIC_RUNTIME_VERSIONS["numpy"] or
            pa.__version__ != PUBLIC_RUNTIME_VERSIONS["pyarrow"]):
        raise ValueError("install NumPy 2.2.6 and PyArrow 25.0.0 for this pinned rebuild")
    return np, pa, pq


def checked_rows(parquet_file, columns, budget):
    for batch in parquet_file.iter_batches(batch_size=4096, columns=list(columns), use_threads=False):
        budget.check()
        names = batch.schema.names
        if tuple(names) != tuple(columns):
            raise ValueError("projected column order mismatch")
        values = [batch.column(i).to_pylist() for i in range(len(columns))]
        for row in zip(*values):
            yield dict(zip(names, row))
        budget.check()


def build(args):
    budget = Budget()
    np, pa, pq = verify_runtime()
    paths = {"spectra": args.spectra, "structures": args.structures, "mass": args.mass}
    for role, path in paths.items():
        size, digest = PINS[role]
        if path.stat().st_size != size or sha256(path) != digest:
            raise ValueError("exact source pin mismatch: " + role)
        budget.check()
    spectra = pq.ParquetFile(args.spectra)
    structures = pq.ParquetFile(args.structures)
    if spectra.metadata.num_rows != 139744 or structures.metadata.num_rows != 20204:
        raise ValueError("source row count differs from pinned source card/audit")
    for name in SPECTRA_COLUMNS:
        field = spectra.schema_arrow.field(name)
        if not (pa.types.is_string(field.type) or pa.types.is_large_string(field.type)):
            raise ValueError("source spectrum metadata type mismatch: " + name)
    for name in STRUCTURE_COLUMNS:
        field = structures.schema_arrow.field(name)
        if name == "inchikey" and not (pa.types.is_string(field.type) or pa.types.is_large_string(field.type)):
            raise ValueError("source structure key type mismatch")
        if name == "exact_mass" and not pa.types.is_float64(field.type):
            raise ValueError("source exact_mass must be float64")
    mass = np.load(args.mass, mmap_mode="r", allow_pickle=False)
    if mass.shape != (436389,) or mass.dtype != np.dtype("<f8"):
        raise ValueError("COCONUT mass header mismatch")
    previous = None
    for start in range(0, len(mass), 65536):
        chunk = mass[start:start + 65536]
        if not np.isfinite(chunk).all() or (chunk <= 0).any() or (chunk[1:] < chunk[:-1]).any():
            raise ValueError("COCONUT mass numeric contract mismatch")
        if previous is not None and chunk[0] < previous:
            raise ValueError("mass sorting differs across chunks")
        previous = chunk[-1]
        budget.check()
    args.out.mkdir()
    payload = args.out / "payload"
    payload.mkdir()
    audit = Counter()
    licences = Counter()
    references = defaultdict(lambda: {"occurrences": 0, "ids": set(), "institutions": set(), "conditions": set()})
    conditions = Counter()
    supplied_id_first_key = {}
    with (payload / "record_attribution.jsonl").open("xb") as stream:
        for source_row, row in enumerate(checked_rows(spectra, SPECTRA_COLUMNS, budget)):
            licences[row["license"]] += 1
            if row["license"] != "CC0":
                continue
            audit["admitted_spectra"] += 1
            if not key_valid(row["inchikey"]):
                audit["malformed_admitted_keys"] += 1
            ref = references[row["inchikey"]]
            ref["occurrences"] += 1
            supplied_id = row["spectrum_id"]
            id_present = isinstance(supplied_id, str) and supplied_id != ""
            repeated_id = id_present and supplied_id in supplied_id_first_key
            if id_present:
                ref["ids"].add(supplied_id)
                audit["repeated_supplied_id_occurrences"] += int(repeated_id)
                if repeated_id and supplied_id_first_key[supplied_id] != row["inchikey"]:
                    audit["repeated_supplied_id_with_different_raw_key"] += 1
                if not repeated_id:
                    supplied_id_first_key[supplied_id] = row["inchikey"]
            else:
                audit["missing_supplied_id_occurrences"] += 1
            if row["source_institution"] not in (None, ""):
                ref["institutions"].add(row["source_institution"])
            condition = (row["inchikey"], row["ion_mode"], row["adduct"], row["collision_energy"], row["source_institution"])
            ref["conditions"].add(condition)
            conditions[condition] += 1
            budget.write_json(stream, {"source_row_id": source_row, **row,
                                      "full_inchikey_valid": key_valid(row["inchikey"]),
                                      "supplied_spectrum_id_present": id_present,
                                      "supplied_spectrum_id_seen_before": repeated_id}, line=True)
    if audit["admitted_spectra"] != 6951:
        raise ValueError("CC0-labelled row count differs from pinned source audit")
    by_key = defaultdict(list)
    for source_row, row in enumerate(checked_rows(structures, STRUCTURE_COLUMNS, budget)):
        if key_valid(row["inchikey"]) and row["inchikey"] in references:
            by_key[row["inchikey"]].append((source_row, row["exact_mass"]))
    real_scan_controls = 0
    with (payload / "reference_molecules.jsonl").open("xb") as stream:
        for key, rows in by_key.items():
            ref = references[key]
            source_mass_values = {repr(value) for _, value in rows}
            conflict = len(source_mass_values) > 1
            for source_row, value in rows:
                issue = "null" if value is None else "non_finite" if not math.isfinite(value) else "non_positive" if value <= 0 else None
                row = {"inchikey": key, "full_inchikey_valid": key_valid(key),
                       "source_structure_row_id": source_row,
                       "supplied_exact_mass": float(value) if issue is None else None,
                       "supplied_exact_mass_source_repr": repr(value),
                       "supplied_mass_issue": issue, "reference_neutral_mass_basis_verified": False,
                       "source_key_row_multiplicity": len(rows), "source_key_mass_conflict": conflict,
                       "admitted_spectra_occurrences": ref["occurrences"],
                       "distinct_admitted_spectrum_ids": len(ref["ids"]),
                       "distinct_nonempty_admitted_institutions": len(ref["institutions"]),
                       "distinct_admitted_raw_conditions": len(ref["conditions"])}
                counts = []
                for ppm in TOLERANCES:
                    count = None
                    if issue is None:
                        lower, upper = mass_bounds(float(value), ppm)
                        left = int(np.searchsorted(mass, lower, side="left"))
                        right = int(np.searchsorted(mass, upper, side="right"))
                        count = right - left
                        if real_scan_controls < 30:
                            direct = int(np.count_nonzero((mass >= lower) & (mass <= upper)))
                            if direct != count:
                                raise ValueError("real direct-scan window control differs")
                            real_scan_controls += 1
                    counts.append(count)
                    row["coconut_row_count_at_supplied_mass_" + str(int(ppm)) + "ppm"] = count
                if issue is None and counts != sorted(counts):
                    raise ValueError("candidate count monotonicity failed")
                budget.write_json(stream, row, line=True)
                audit["molecule_output_rows"] += 1
                audit["molecule_rows_with_invalid_supplied_mass"] += int(issue is not None)
                audit["molecule_rows_with_key_mass_conflict"] += int(conflict)
    with (payload / "condition_multiplicity.jsonl").open("xb") as stream:
        for condition, count in conditions.items():
            row = dict(zip(("inchikey", "ion_mode", "adduct", "collision_energy", "source_institution"), condition))
            budget.write_json(stream, {**row, "admitted_source_occurrences": count}, line=True)
    unmatched = set(references) - set(by_key)
    with (payload / "unmatched_reference_keys.jsonl").open("xb") as stream:
        for key in sorted(unmatched, key=lambda value: (value is not None, value or "")):
            budget.write_json(stream, {"inchikey": key, "full_inchikey_valid": key_valid(key),
                                      "admitted_source_occurrences": references[key]["occurrences"]}, line=True)
    manifest = {"kind": "public_reference_evidence_supplied_mass_windows",
                "licence_rule": 'exact raw spectrum license == "CC0"',
                "derivative_compilation_scope": "CC-BY-NC-SA-4.0; retain original record attribution and terms",
                "mass_basis": "supplied reference exact_mass; neutral basis unqualified",
                "source_dataset_ref": "samartalwar/casmi-2026-spectral-library-massbankharmonized",
                "source_dataset_numeric_id": 12067949,
                "sources": {role: {"bytes": value[0], "sha256": value[1]} for role, value in PINS.items()},
                "admitted_spectra_occurrences": audit["admitted_spectra"],
                "admitted_distinct_raw_keys": len(references), "unmatched_raw_keys": len(unmatched),
                "malformed_admitted_keys": audit["malformed_admitted_keys"],
                "missing_supplied_id_occurrences": audit["missing_supplied_id_occurrences"],
                "repeated_supplied_id_occurrences": audit["repeated_supplied_id_occurrences"],
                "repeated_supplied_id_with_different_raw_key": audit["repeated_supplied_id_with_different_raw_key"],
                "molecule_output_rows": audit["molecule_output_rows"],
                "real_direct_scan_controls": real_scan_controls,
                "ppm_values": list(TOLERANCES), "closed_bounds": True, "candidate_cap": None,
                "casmi_membership_fields_read": False, "competition_2026_files_read": False,
                "chemical_training_overlap_measured": False, "model_execution": False,
                "structural_overlap_verified": False, "official_or_model_score": None,
                "runtime_versions": {"python": platform.python_version(), "numpy": np.__version__, "pyarrow": pa.__version__}}
    with (payload / "MANIFEST.json").open("xb") as stream:
        budget.write_json(stream, manifest)
    files = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(payload.iterdir())}
    budget.check()
    receipt = {"status": "ACTUAL_PORTABLE_REBUILD_COMPLETE", "observed_at_utc": datetime.now(timezone.utc).isoformat(),
               "builder_sha256": sha256(Path(__file__)), "runtime_versions": manifest["runtime_versions"],
               "wall_seconds": time.monotonic() - budget.started,
               "cpu_seconds": time.process_time() - budget.cpu_started, "max_rss_bytes": budget.rss(),
               "payload_bytes": budget.bytes_written, "files": files,
               "anomaly_counts": dict(audit), "notebook_operations": False, "network_sdk_or_submission": False}
    with (args.out / "REBUILD_RECEIPT.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    signal.setitimer(signal.ITIMER_REAL, 0)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("spectra", "structures", "mass", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = build(args)
    print(json.dumps({"status": result["status"], "payload_bytes": result["payload_bytes"],
                      "wall_seconds": result["wall_seconds"], "max_rss_bytes": result["max_rss_bytes"]}))


if __name__ == "__main__":
    main()
