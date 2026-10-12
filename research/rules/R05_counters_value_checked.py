ID = "R05"
TAGS = ["g4", "decision"]
LESSON = "A pushed probe has at least one counter whose printed value must be > 0; presence alone proves nothing (L05)."
SRC = "20261010-adduct-iceberg-rerank: G4 PASS on a log line saying `molecules reranked 0`"
FLAGS = ["rules/fixtures/R05_presence_only"]

from _util import text  # noqa: E402


def check(idea):
    c = [x for x in text(idea, "kaggle_push/counters.txt").splitlines() if x.strip()]
    if (idea / "kaggle_push").exists() and "kind: control" not in text(idea, "STATUS") and not [x for x in c if not x.startswith("?")]:
        return ["kaggle_push/counters.txt: no value-checked counter (all missing or `?`-prefixed)"]
    return []
