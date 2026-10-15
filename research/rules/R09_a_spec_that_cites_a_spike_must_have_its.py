ID = "R09"
TAGS = ["spec", "spike"]
LESSON = "A spec that cites a spike log must have spike_result.md filled (question, script, numbers, verdict), not the blank template."
SRC = ("2026-10-14 formula-window: the spec's premise check ran a spike in scratch/ and cited its log, "
       "while the idea folder's spike_result.md stayed the blank template")
FLAGS = ["rules/fixtures/R09_spike_unfilled", "exploit/20261014-formula-window"]

import re  # noqa: E402
from _util import text  # noqa: E402


def check(idea):
    spec = text(idea, "spec.md")
    res = text(idea, "spike_result.md")
    if not re.search(r"\bspike(?!_result)", spec, re.I) or not res:  # the spec template names spike_result.md
        return []
    if "<one line>" in res or "<cached arrays / cohort used>" in res:
        return ["spike_result.md: spec.md cites a spike but spike_result.md is still the blank template"]
    return []
