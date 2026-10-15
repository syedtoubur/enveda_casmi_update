"""Original standard-library readers for the public reference evidence atlas."""
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 cubres
from pathlib import Path
import hashlib
import json
import re

TABLES = frozenset({"record_attribution.jsonl", "reference_molecules.jsonl", "condition_multiplicity.jsonl"})
KEY = re.compile(r"[A-Z]{14}-[A-Z]{10}-[A-Z]\Z")


def _table(directory, filename):
    if filename not in TABLES:
        raise ValueError("unknown atlas table")
    with (Path(directory) / filename).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError("blank JSONL line at " + str(line_number))
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("table row must be a JSON object")
            yield row


def read_reference_records(directory):
    """Yield every source structure row; never deduplicate or normalize keys."""
    yield from _table(directory, "reference_molecules.jsonl")


def evidence_for_key(directory, key):
    """Yield all admitted reference occurrences for an exact valid full key."""
    if not isinstance(key, str) or KEY.fullmatch(key) is None:
        raise ValueError("provide an exact full 27-character supplied key")
    for row in _table(directory, "record_attribution.jsonl"):
        if row.get("inchikey") == key:
            yield row


def conditions_for_key(directory, key):
    """Yield raw condition groups, retaining missing values and source strings."""
    if not isinstance(key, str) or KEY.fullmatch(key) is None:
        raise ValueError("provide an exact full 27-character supplied key")
    for row in _table(directory, "condition_multiplicity.jsonl"):
        if row.get("inchikey") == key:
            yield row


def verify_downloadable_files(directory):
    """Verify the flat public inventory; no private build envelope is needed."""
    directory = Path(directory)
    inventory = json.loads((directory / "RELEASE_FILES.json").read_text(encoding="utf-8"))
    if inventory.get("schema") != "release-file-hashes-v1" or not isinstance(inventory.get("files"), dict):
        raise ValueError("unsupported public inventory")
    for name, expected in inventory["files"].items():
        if not isinstance(name, str) or Path(name).name != name or "/" in name or "\\" in name or name in (".", ".."):
            raise ValueError("inventory must contain flat file names")
        path = directory / name
        if path.is_symlink():
            raise ValueError("downloadable file must not be a symlink: " + name)
        if path.stat().st_size != expected["bytes"]:
            raise ValueError("file size differs: " + name)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected["sha256"]:
            raise ValueError("file hash differs: " + name)
    return {"status": "PASS_PUBLIC_FILE_PINS", "verified_files": len(inventory["files"])}
