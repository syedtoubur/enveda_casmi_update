ID = "R01"
TAGS = ["spike", "measurement"]
LESSON = "Counts from local test.parquet or a draft run are placeholder data (a train slice), never the hidden test (L04)."
SRC = "20261010-adduct-iceberg-rerank: '34 of 400 test molecules' counted on test.parquet, passed critic and judge"
FLAGS = ["exploit/20261010-adduct-iceberg-rerank"]

import re  # noqa: E402
from _util import lines  # noqa: E402


def check(idea):
    out = []
    for f, i, ln in lines(idea, "spike_result.md", "spec.md", "hypothesis.md"):
        if re.search(r"test\.parquet", ln) and not re.search(r"placeholder|L04", ln, re.I):
            out.append(f"{f}:{i}: test.parquet used without saying it is placeholder data (L04)")
    return out
